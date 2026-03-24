import os
import sys
import csv
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LabeledMemmapDataset, ChunkedRandomSampler
from smrt_foundation.model import DirectClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import ZNorm

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EXP_DIR, 'config.yaml')

DEFAULT = {
    'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 128,
    'batch_size': 64, 'epochs': 10, 'ds_limit': 2_000_000,
    'max_lr': 1e-3, 'weight_decay': 0.02, 'pct_start': 0.4,
}


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config_updated = DEFAULT | config.get('classifier', {})
    config['classifier'] = config_updated

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='no',
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        print(config)

    set_seed(42)

    tmp_ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=config_updated['ds_limit'])
    train_norm_fn = ZNorm(tmp_ds)

    train_ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=config_updated['ds_limit'], norm_fn=train_norm_fn, balance=True)
    train_sampler = ChunkedRandomSampler(train_ds, 2048, shuffle_within=False)
    train_dl = DataLoader(train_ds, batch_size=config_updated['batch_size'], num_workers=4, pin_memory=True, prefetch_factor=4, sampler=train_sampler)

    model = DirectClassifier(
        d_model=config_updated['d_model'],
        n_layers=config_updated['n_layers'],
        n_head=config_updated['n_head'],
        max_len=config_updated['context']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config_updated['max_lr']), weight_decay=config_updated['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    total_steps = len(train_dl) * config_updated['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=config_updated['pct_start']
    )
    scheduler = accelerator.prepare(scheduler)

    # diagnostics file
    diag_path = os.path.join(EXP_DIR, 'diagnostics.csv')
    diag_file = None
    diag_writer = None
    if accelerator.is_main_process:
        diag_file = open(diag_path, 'w', newline='')
        diag_writer = csv.writer(diag_file)
        diag_writer.writerow(['step', 'epoch', 'data_time', 'forward_time', 'backward_time', 'step_time', 'loss'])

    global_step = 0

    for epoch in range(config_updated['epochs']):
        model.train()
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config_updated['epochs']}", disable=not accelerator.is_main_process)

        t_data_start = time.perf_counter()
        for x, y in progress_bar:
            t_data_end = time.perf_counter()
            data_time = t_data_end - t_data_start

            # forward
            t_fwd_start = time.perf_counter()
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()
            forward_time = t_fwd_end - t_fwd_start

            # backward
            t_bwd_start = time.perf_counter()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_bwd_end = time.perf_counter()
            backward_time = t_bwd_end - t_bwd_start

            step_time = data_time + forward_time + backward_time
            global_step += 1
            loss_val = loss.item()

            if accelerator.is_main_process:
                diag_writer.writerow([global_step, epoch, f'{data_time:.4f}', f'{forward_time:.4f}', f'{backward_time:.4f}', f'{step_time:.4f}', f'{loss_val:.4f}'])
                progress_bar.set_postfix(loss=f"{loss_val:.4f}", data=f"{data_time:.3f}", fwd=f"{forward_time:.3f}", bwd=f"{backward_time:.3f}")

            t_data_start = time.perf_counter()

    if diag_file:
        diag_file.close()

    if accelerator.is_main_process:
        print(f"\nDiagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
