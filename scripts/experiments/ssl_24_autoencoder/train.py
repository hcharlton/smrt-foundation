"""
Masked autoencoder pretraining.

Reconstructs masked kinetics directly (MSE loss) instead of contrastive
matching (InfoNCE). The encoder must preserve actual kinetics information
in its representations to enable reconstruction.
"""

import sys
import os
import subprocess
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import ShardedMemmapDataset, LabeledMemmapDataset
from smrt_foundation.model import SmrtAutoencoder
from smrt_foundation.loss import MaskedReconstructionLoss
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def linear_probe_eval(encoder, probe_config, config, accelerator, norm_fn):
    """Freeze encoder, train a linear head on labeled CpG data, report accuracy."""
    device = accelerator.device
    encoder.eval()

    pc = probe_config
    probe_limit = pc.get('ds_limit', 500000)

    train_ds = LabeledMemmapDataset(
        config.get('probe_pos_train'), config.get('probe_neg_train'),
        limit=probe_limit, norm_fn=norm_fn, balance=True
    )
    val_ds = LabeledMemmapDataset(
        config.get('probe_pos_val'), config.get('probe_neg_val'),
        limit=probe_limit, norm_fn=norm_fn
    )

    train_dl = DataLoader(train_ds, batch_size=pc.get('batch_size', 512), shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=pc.get('batch_size', 512), shuffle=False, num_workers=2)

    d_model = encoder.d_model
    probe_head = nn.Linear(d_model, 1).to(device)
    probe_opt = torch.optim.Adam(probe_head.parameters(), lr=float(pc.get('lr', 3e-3)))
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(pc.get('epochs', 3)):
        probe_head.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                c = encoder.forward(x)
                center = c[:, c.shape[1] // 2, :]
            logits = probe_head(center).squeeze(-1)
            loss = criterion(logits, y)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

    probe_head.eval()
    acc_metric = BinaryAccuracy().to(device)
    auroc_metric = BinaryAUROC().to(device)

    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            c = encoder.forward(x)
            center = c[:, c.shape[1] // 2, :]
            logits = probe_head(center).squeeze(-1)
        acc_metric.update(logits > 0, y.long())
        auroc_metric.update(logits, y.long())

    top1 = acc_metric.compute().item()
    auroc = auroc_metric.compute().item()

    del probe_head, probe_opt, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache()

    return top1, auroc


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 128,
        'batch_size': 512, 'epochs': 12, 'ds_limit': 2_000_000,
        'max_lr': 3e-4, 'p_mask': 0.15, 'mask_size': 10,
        'weight_decay': 0.02, 'pct_start': 0.25,
    }

    c = DEFAULT | config.get('autoencoder', {})
    config['autoencoder'] = c
    config['git_hash'] = get_git_revision_hash()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="tensorboard",
        project_dir="training_logs",
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        print(config)

    set_seed(42)

    exp_type = config.get('experiment_type', 'ssl')
    exp_name = config.get('experiment_name', 'ssl_experiment')
    project_namespace = f"{exp_type}/{exp_name}"

    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker("tensorboard")
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, "hparams.yaml"), "w") as f:
            yaml.dump(config, f)
        tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # Dataset with normalization
    dataset_name = config.get('ssl_dataset', 'ob007_raw.memmap')
    memmap_path = f"data/01_processed/ssl_sets/{dataset_name}"

    ds = ShardedMemmapDataset(memmap_path, limit=c['ds_limit'])

    if accelerator.is_main_process:
        print(f"SSL dataset: {len(ds)} samples from {memmap_path}")

    ssl_norm = KineticsNorm(ds, max_samples=16_384)
    if accelerator.is_main_process:
        print(f"SSL norm — means: {ssl_norm.means}, stds: {ssl_norm.stds}")

    class NormedDataset(torch.utils.data.Dataset):
        def __init__(self, inner, norm_fn, context=None):
            self.inner = inner
            self.norm_fn = norm_fn
            self.context = context
        def __len__(self):
            return len(self.inner)
        def __getitem__(self, idx):
            x = self.inner[idx]
            if self.context is not None:
                x = x[:self.context]
            return self.norm_fn(x)

    normed_ds = NormedDataset(ds, ssl_norm, context=c['context'])
    dl = DataLoader(normed_ds, batch_size=c['batch_size'], num_workers=2,
                    pin_memory=True, prefetch_factor=4, shuffle=True)

    model = SmrtAutoencoder(
        d_model=c['d_model'], n_layers=c['n_layers'], n_head=c['n_head'],
        max_len=c['context'], p_mask=c['p_mask'], mask_size=c['mask_size']
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay'])
    criterion = MaskedReconstructionLoss()

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    total_steps = len(dl) * c['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps=total_steps, pct_start=c['pct_start'])
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(dl)}, Total steps: {total_steps}")

    checkpoint_dir = os.path.join(os.path.dirname(config_path), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0

    for epoch in range(c['epochs']):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{c['epochs']}",
                           disable=not accelerator.is_main_process)

        for batch in progress_bar:
            kin_recon, kin_target, mask = model(batch)
            loss = criterion(kin_recon, kin_target, mask)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            global_step += 1
            current_loss = accelerator.reduce(loss, reduction="mean").item()
            epoch_loss += current_loss

            accelerator.log({
                "train_loss": current_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=global_step)

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{current_loss:.4f}")

        avg_loss = epoch_loss / len(dl)
        accelerator.log({"epoch_avg_loss": avg_loss}, step=global_step)

        # --- Linear probe evaluation ---
        probe_config = config.get('probe', {})
        if config.get('probe_pos_train'):
            unwrapped = accelerator.unwrap_model(model)
            probe_top1, probe_auroc = linear_probe_eval(
                unwrapped.encoder, probe_config, config, accelerator, ssl_norm
            )
            if accelerator.is_main_process:
                accelerator.log({
                    "probe_top1": probe_top1,
                    "probe_auroc": probe_auroc,
                }, step=global_step)
                print(f"Epoch {epoch+1}: recon_loss={avg_loss:.4f}  probe_top1={probe_top1:.4f}  probe_auroc={probe_auroc:.4f}")
        elif accelerator.is_main_process:
            print(f"Epoch {epoch+1}: recon_loss={avg_loss:.4f}")

        accelerator.wait_for_everyone()
        model.train()

    # --- Save final checkpoint ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': c['epochs'],
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
