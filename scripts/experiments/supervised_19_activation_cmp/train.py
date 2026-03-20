"""
Activation comparison experiment.

Trains DirectClassifier on both legacy (parquet) and new (memmap) datasets,
then collects encoder center-position activations and logits from eval data.
Saves results as .npz for the plotting script at report/eda/activation_comparison/plot.py.
"""

import sys
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import (
    LabeledMemmapDataset, LegacyMethylDataset, compute_log_normalization_stats
)
from smrt_foundation.model import DirectClassifier
from smrt_foundation.normalization import ZNorm
from smrt_foundation.optim import get_cosine_schedule_with_warmup


def train_model(model, train_dl, epochs, lr, weight_decay, pct_start, device):
    """Train a DirectClassifier and return the trained model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    total_steps = len(train_dl) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps=total_steps, pct_start=pct_start)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_dl, desc=f"  Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return model


@torch.no_grad()
def collect_activations(model, eval_dl, max_samples, device):
    """Run model on eval data, collect center activations and logits."""
    model.eval()
    all_acts = []
    all_logits = []
    all_labels = []
    n = 0

    for x, y in eval_dl:
        if n >= max_samples:
            break
        x = x.to(device)
        # Get encoder output (before classification head)
        c = model.encoder.forward(x)  # (B, T_down, d_model)
        center = c[:, c.shape[1] // 2, :]  # (B, d_model)
        logits = model.head(center)  # (B, 1)

        all_acts.append(center.cpu().numpy())
        all_logits.append(logits.squeeze(-1).cpu().numpy())
        all_labels.append(y.numpy())
        n += x.shape[0]

    return (
        np.concatenate(all_acts)[:max_samples],
        np.concatenate(all_logits)[:max_samples],
        np.concatenate(all_labels)[:max_samples],
    )


def make_legacy_dataloader(parquet_path, context, batch_size, ds_limit):
    """Create a DataLoader from legacy parquet, returning (x, y) tuples."""
    df = pl.read_parquet(parquet_path).head(ds_limit)
    kin_feats = ['fi', 'fp', 'ri', 'rp']
    means, stds = compute_log_normalization_stats(df, kin_feats)

    ds = LegacyMethylDataset(
        parquet_path, means, stds, context,
        restrict_row_groups=100, single_strand=True
    )

    # Wrap IterableDataset to yield (x, y) tuples instead of dicts
    class LegacyWrapper(torch.utils.data.IterableDataset):
        def __init__(self, inner):
            self.inner = inner
        def __iter__(self):
            for item in self.inner:
                yield item['data'], item['label'].float()
        def __len__(self):
            return len(self.inner)

    wrapped = LegacyWrapper(ds)
    return DataLoader(wrapped, batch_size=batch_size, drop_last=True)


def make_new_dataloader(pos_path, neg_path, batch_size, ds_limit):
    """Create a DataLoader from new memmap pipeline."""
    tmp_ds = LabeledMemmapDataset(pos_path, neg_path, limit=ds_limit)
    norm_fn = ZNorm(tmp_ds, log_transform=True)
    ds = LabeledMemmapDataset(pos_path, neg_path, limit=ds_limit, norm_fn=norm_fn, balance=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    c = config['classifier']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    output_dir = os.path.dirname(config_path)
    npz_path = os.path.join(output_dir, 'activations.npz')

    print("=" * 60)
    print("ACTIVATION COMPARISON EXPERIMENT")
    print("=" * 60)

    # --- Train on legacy ---
    print("\n[1/4] Loading legacy dataset...")
    legacy_train_dl = make_legacy_dataloader(
        config['legacy_train'], c['context'], c['batch_size'], c['ds_limit']
    )
    legacy_val_dl = make_legacy_dataloader(
        config['legacy_val'], c['context'], c['batch_size'], c['ds_limit']
    )

    print("[2/4] Training on LEGACY data...")
    legacy_model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context']
    ).to(device)
    legacy_model = train_model(
        legacy_model, legacy_train_dl, c['epochs'],
        float(c['max_lr']), c['weight_decay'], c['pct_start'], device
    )

    print("  Collecting legacy activations...")
    leg_acts, leg_logits, leg_labels = collect_activations(
        legacy_model, legacy_val_dl, c.get('eval_samples', 20000), device
    )

    del legacy_model, legacy_train_dl, legacy_val_dl
    torch.cuda.empty_cache()

    # --- Train on new ---
    print("\n[3/4] Loading new dataset...")
    new_train_dl = make_new_dataloader(
        config['pos_data_train'], config['neg_data_train'],
        c['batch_size'], c['ds_limit']
    )
    new_val_dl = make_new_dataloader(
        config['pos_data_val'], config['neg_data_val'],
        c['batch_size'], c['ds_limit']
    )

    print("[4/4] Training on NEW data...")
    new_model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context']
    ).to(device)
    new_model = train_model(
        new_model, new_train_dl, c['epochs'],
        float(c['max_lr']), c['weight_decay'], c['pct_start'], device
    )

    print("  Collecting new activations...")
    new_acts, new_logits, new_labels = collect_activations(
        new_model, new_val_dl, c.get('eval_samples', 20000), device
    )

    # --- Save ---
    np.savez(
        npz_path,
        legacy_activations=leg_acts,
        legacy_logits=leg_logits,
        legacy_labels=leg_labels,
        new_activations=new_acts,
        new_logits=new_logits,
        new_labels=new_labels,
    )
    print(f"\nSaved activations to {npz_path}")
    print(f"  Legacy: {leg_acts.shape[0]} samples, {leg_acts.shape[1]} dims")
    print(f"  New:    {new_acts.shape[0]} samples, {new_acts.shape[1]} dims")


if __name__ == "__main__":
    main()
