import sys
import os
import subprocess
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

module_path = os.path.abspath("/dcai/users/chache/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import ShardedMemmapDataset
from smrt_foundation.model import Smrt2Vec
from smrt_foundation.loss import InfoNCE
from smrt_foundation.optim import get_cosine_schedule_with_warmup

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT_SMRT2VEC = {
        'd_model': 128,
        'n_layers': 4,
        'n_head': 4,
        'context': 4096,
        'batch_size': 64,
        'epochs': 10,
        'max_lr': 3e-4,
        'temperature': 0.1,
        'p_mask': 0.05,
        'weight_decay': 0.02,
        'pct_start': 0.25
    }

    config_updated = DEFAULT_SMRT2VEC | config.get('smrt2vec', {})
    config['smrt2vec'] = config_updated
    config['git_hash'] = get_git_revision_hash()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='no',
        log_with="tensorboard",
        project_dir="training_logs",
        kwargs_handlers=[ddp_kwargs]
    )

    set_seed(42)

    exp_type = config.get('experiment_type', 'ssl')
    exp_name = config.get('experiment_name', 'smrt_experiment')
    project_namespace = f"{exp_type}/{exp_name}"

    if accelerator.is_main_process:
            accelerator.init_trackers(project_namespace)
            
            tracker = accelerator.get_tracker("tensorboard")
            run_dir = tracker.writer.log_dir
            
            with open(os.path.join(run_dir, "hparams.yaml"), "w") as f:
                yaml.dump(config, f)
                
            tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    dataset_name = config.get('ssl_dataset', 'ob007')
    memmap_path = f"data/01_processed/ssl_sets/{dataset_name}.memmap"
    
    ds = ShardedMemmapDataset(memmap_path, limit=2_000_000)
    dl = DataLoader(
        ds,
        batch_size=config_updated['batch_size'],
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True
    )

    model = Smrt2Vec(
        d_model=config_updated['d_model'],
        n_layers=config_updated['n_layers'],
        n_head=config_updated['n_head'],
        max_len=config_updated['context'],
        p_mask=config_updated['p_mask']
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config_updated['max_lr']),
        weight_decay=config_updated['weight_decay']
    )
    criterion = InfoNCE(temperature=float(config_updated['temperature']))

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        total_steps=len(dl) * config_updated['epochs'],
        pct_start=config_updated['pct_start']
    )
    scheduler = accelerator.prepare(scheduler)

    global_step = 0

    for epoch in range(config_updated['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{config_updated['epochs']}") if accelerator.is_main_process else dl
            
        for batch in progress_bar:
            c_proj, targets, mask_idx = model(batch)
            loss = criterion(c_proj, targets, mask_idx)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            current_loss = loss.item()
            epoch_loss += current_loss

            accelerator.log({
                "train_loss": current_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=global_step)

            if accelerator.is_main_process:
                 progress_bar.set_postfix(loss=f"{current_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(dl)
        accelerator.log({"epoch_avg_loss": avg_epoch_loss}, step=global_step)
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    accelerator.end_training()

if __name__ == "__main__":
    main()