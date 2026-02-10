import sys
import os
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

def main():
    BATCH_SIZE = 64
    EPOCHS = 12
    LEARNING_RATE = 3e-4
    SEED = 42
    OUTPUT_DIR = "training_logs" 

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        # mixed_precision="bf16", 
        mixed_precision='no',
        log_with="tensorboard",
        project_dir=OUTPUT_DIR,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("smrt_experiment_04", config={
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "d_model": 256
        })

    set_seed(SEED)

    ds = ShardedMemmapDataset('data/01_processed/ssl_sets/ob007.memmap', limit=2_000_000) 
    dl = DataLoader(
        ds,  
        batch_size=BATCH_SIZE, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True
    )

    model = Smrt2Vec(d_model=128, n_layers=4, n_head=4, max_len=4096)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
    criterion = InfoNCE(temperature=0.1)

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=len(dl) * EPOCHS, 
        pct_start=0.1,
        div_factor=100,
        final_div_factor=100
    )
    scheduler = accelerator.prepare(scheduler)

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        if accelerator.is_main_process:
            progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        else:
            progress_bar = dl
            
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
            current_lr = scheduler.get_last_lr()[0]

            accelerator.log(
                {
                    "train_loss": current_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch
                }, 
                step=global_step
            )

            if accelerator.is_main_process:
                 progress_bar.set_postfix(loss=f"{current_loss:.4f}")
        avg_epoch_loss = epoch_loss / len(dl)
        accelerator.log(
            {"epoch_avg_loss": avg_epoch_loss},
            step = global_step
        )
        accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    accelerator.end_training()

if __name__ == "__main__":
    main()