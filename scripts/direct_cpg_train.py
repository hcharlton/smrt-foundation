import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import polars as pl

module_path = os.path.abspath("/dcai/users/chache/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, compute_log_normalization_stats
from smrt_foundation.model import DirectClassifier
from smrt_foundation.loss import InfoNCE

def main():
    BATCH_SIZE = 2**13
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    SEED = 42
    OUTPUT_DIR = "training_logs" 

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="bf16", 
        log_with="tensorboard",
        project_dir=OUTPUT_DIR,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("smrt_experiment_01", config={
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "d_model": 256
        })

    set_seed(SEED)


    KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']

    train_df = pl.read_parquet('/tmp/pacbio_standard_train.parquet').head(2_000_000)
    train_means, train_stds = compute_log_normalization_stats(train_df, KINETICS_FEATURES)

    # batch_size=2048
    single_strand=True

    #train
    train_ds = LegacyMethylDataset('/tmp/pacbio_standard_train.parquet',
                                    means=train_means,
                                    stds=train_stds,
                                    context=32,
                                    single_strand=True)
    train_dl = DataLoader(train_ds,
                            batch_size=BATCH_SIZE,
                            drop_last=True,
                            persistent_workers=False,
                            prefetch_factor=None)
    # val
    val_ds = LegacyMethylDataset('/tmp/pacbio_standard_test.parquet',
                                means=train_means,
                                stds=train_stds,
                                context=32,
                                single_strand=True)
    val_dl = DataLoader(val_ds,
                        batch_size=BATCH_SIZE,
                        drop_last=True,
                        persistent_workers=False,
                        prefetch_factor=None)


    model = DirectClassifier(d_model=128, n_layers=4, n_head=4, max_len=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-4,
        total_steps=len(train_dl) * EPOCHS, 
        pct_start=0.05
    )
    scheduler = accelerator.prepare(scheduler)

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        
        if accelerator.is_main_process:
            progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        else:
            progress_bar = train_dl
        ### train loop
        for batch in progress_bar:
            x = batch['data']
            y = batch['label']

            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            global_step += 1

            current_loss = loss.item()
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
        ### eval loop
        model.eval()
        total_count = 0
        total_correct = 0
        progress_bar = tqdm(val_dl, desc=f"Eval set run {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            x = batch['data']
            y = batch['label']

            logits = model(x)
            y_hat = logits>0
            correct = y == y_hat.squeeze(-1)
            total_count += y_hat.shape[0]
            total_correct += correct.sum()
        print(f'Epoch {epoch+1} Val Top1 Accuracy: {total_correct/total_count}')

        

    accelerator.end_training()

if __name__ == "__main__":
    main()