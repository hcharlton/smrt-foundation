"""
Fine-tune a pretrained SSL encoder for CpG methylation classification.

Two-stage training:
  Stage 1: Freeze encoder, train classification head only (warmup)
  Stage 2: Unfreeze encoder, fine-tune all with differential learning rates
           (lower LR for pretrained encoder, higher for head)
"""

import sys
import os
import subprocess
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def load_pretrained_encoder(checkpoint_path, model):
    """Load pretrained encoder weights into DirectClassifier."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_weights = checkpoint['encoder_state_dict']

    # Transfer encoder weights (skip PE buffer size mismatch by loading non-strict)
    missing, unexpected = model.encoder.load_state_dict(encoder_weights, strict=False)

    print(f"Loaded pretrained encoder from {checkpoint_path}")
    if missing:
        print(f"  Missing keys (expected): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")

    return model


def eval_epoch(model, val_dl, accelerator, f1_metric, auroc_metric, auprc_metric, acc_metric):
    """Run evaluation and return metrics."""
    model.eval()
    for x, y in val_dl:
        with torch.no_grad():
            logits = model(x)
        y_hat, y, logits = accelerator.gather_for_metrics((logits > 0, y, logits))
        f1_metric.update(y_hat.squeeze(-1), y.long())
        auroc_metric.update(logits.squeeze(-1), y.long())
        auprc_metric.update(logits.squeeze(-1), y.long())
        acc_metric.update(y_hat.squeeze(-1), y.long())

    results = {
        'f1': f1_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'auprc': auprc_metric.compute().item(),
        'top1': acc_metric.compute().item(),
    }
    f1_metric.reset(); auroc_metric.reset(); auprc_metric.reset(); acc_metric.reset()
    return results


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    c = config.get('classifier', {})
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

    exp_type = config.get('experiment_type', 'supervised')
    exp_name = config.get('experiment_name', 'finetune_experiment')
    project_namespace = f"{exp_type}/{exp_name}"

    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker("tensorboard")
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, "hparams.yaml"), "w") as f:
            yaml.dump(config, f)
        tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # --- Data ---
    norm_limit = min(c.get('ds_limit', 0), 2_000_000) if c.get('ds_limit', 0) > 0 else 2_000_000
    tmp_ds = LabeledMemmapDataset(config['pos_data_train'], config['neg_data_train'], limit=norm_limit)
    train_norm_fn = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=c.get('ds_limit', 0), norm_fn=train_norm_fn, balance=True
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=c.get('ds_limit', 0), norm_fn=train_norm_fn
    )

    train_dl = DataLoader(train_ds, batch_size=c['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=c['batch_size'], num_workers=2, pin_memory=True, shuffle=False)

    if accelerator.is_main_process:
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # --- Model with pretrained weights ---
    model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context']
    )

    checkpoint_path = config.get('pretrained_checkpoint')
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_pretrained_encoder(checkpoint_path, model)
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path} — training from scratch")

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()

    f1_metric, auroc_metric, auprc_metric, acc_metric = accelerator.prepare(
        BinaryF1Score(), BinaryAUROC(), BinaryAveragePrecision(), BinaryAccuracy()
    )

    global_step = 0

    # ==================== STAGE 1: Frozen encoder, train head only ====================
    frozen_epochs = c.get('frozen_epochs', 5)
    frozen_lr = float(c.get('frozen_lr', 3e-3))

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Frozen encoder, head warmup ({frozen_epochs} epochs, lr={frozen_lr})")
        print(f"{'='*60}")

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    head_optimizer = torch.optim.AdamW(model.head.parameters(), lr=frozen_lr, weight_decay=c['weight_decay'])
    model, head_optimizer, train_dl, val_dl = accelerator.prepare(model, head_optimizer, train_dl, val_dl)

    total_frozen_steps = len(train_dl) * frozen_epochs
    head_scheduler = get_cosine_schedule_with_warmup(head_optimizer, total_steps=total_frozen_steps, pct_start=0.1)
    head_scheduler = accelerator.prepare(head_scheduler)

    for epoch in range(frozen_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"S1 Epoch {epoch+1}/{frozen_epochs}", disable=not accelerator.is_main_process)

        for x, y in pbar:
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).float())
            head_optimizer.zero_grad()
            accelerator.backward(loss)
            head_optimizer.step()
            head_scheduler.step()

            global_step += 1
            loss_val = accelerator.reduce(loss, reduction="mean").item()
            epoch_loss += loss_val
            accelerator.log({"train_loss": loss_val, "learning_rate": head_scheduler.get_last_lr()[0], "epoch": epoch, "stage": 1}, step=global_step)
            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = epoch_loss / len(train_dl)
        metrics = eval_epoch(model, val_dl, accelerator, f1_metric, auroc_metric, auprc_metric, acc_metric)
        accelerator.log({"epoch_avg_loss": avg_loss, "eval_f1": metrics['f1'], "eval_auroc": metrics['auroc'], "eval_auprc": metrics['auprc'], "eval_top1": metrics['top1']}, step=global_step)

        if accelerator.is_main_process:
            print(f"S1 Epoch {epoch+1}: loss={avg_loss:.4f}  top1={metrics['top1']:.4f}  auroc={metrics['auroc']:.4f}")

    # ==================== STAGE 2: Unfreeze encoder, fine-tune all ====================
    finetune_epochs = c.get('finetune_epochs', 15)
    encoder_lr = float(c.get('encoder_lr', 3e-4))
    head_lr = float(c.get('head_lr', 3e-3))

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Full fine-tuning ({finetune_epochs} epochs, encoder_lr={encoder_lr}, head_lr={head_lr})")
        print(f"{'='*60}")

    # Unfreeze encoder
    for param in model.parameters():
        param.requires_grad = True

    # Differential learning rates
    unwrapped = accelerator.unwrap_model(model)
    param_groups = [
        {'params': unwrapped.encoder.parameters(), 'lr': encoder_lr},
        {'params': unwrapped.head.parameters(), 'lr': head_lr},
    ]
    ft_optimizer = torch.optim.AdamW(param_groups, weight_decay=c['weight_decay'])
    ft_optimizer = accelerator.prepare(ft_optimizer)

    total_ft_steps = len(train_dl) * finetune_epochs
    ft_scheduler = get_cosine_schedule_with_warmup(ft_optimizer, total_steps=total_ft_steps, pct_start=c.get('pct_start', 0.1))
    ft_scheduler = accelerator.prepare(ft_scheduler)

    for epoch in range(finetune_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"S2 Epoch {epoch+1}/{finetune_epochs}", disable=not accelerator.is_main_process)

        for x, y in pbar:
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).float())
            ft_optimizer.zero_grad()
            accelerator.backward(loss)
            ft_optimizer.step()
            ft_scheduler.step()

            global_step += 1
            loss_val = accelerator.reduce(loss, reduction="mean").item()
            epoch_loss += loss_val
            accelerator.log({"train_loss": loss_val, "learning_rate": ft_scheduler.get_last_lr()[0], "epoch": frozen_epochs + epoch, "stage": 2}, step=global_step)
            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = epoch_loss / len(train_dl)
        metrics = eval_epoch(model, val_dl, accelerator, f1_metric, auroc_metric, auprc_metric, acc_metric)
        accelerator.log({"epoch_avg_loss": avg_loss, "eval_f1": metrics['f1'], "eval_auroc": metrics['auroc'], "eval_auprc": metrics['auprc'], "eval_top1": metrics['top1']}, step=global_step)

        if accelerator.is_main_process:
            print(f"S2 Epoch {epoch+1}: loss={avg_loss:.4f}  top1={metrics['top1']:.4f}  auroc={metrics['auroc']:.4f}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
