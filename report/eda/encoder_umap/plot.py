"""
UMAP of learned encoder representations for CpG methylation.

Loads the SSL-pretrained encoder from experiment 21, runs inference on
labeled CpG data to extract 128-dim hidden states, then projects with
UMAP. Compares to the raw-kinetics UMAP in cpg_umap/ — if the encoder
has learned the conditional kinetics-given-sequence signal, these
representations should show better class separation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
import umap

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.model import SmrtEncoder
from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.normalization import KineticsNorm


CHECKPOINT = 'scripts/experiments/ssl_21_pretrain/checkpoints/final_model.pt'
POS_TRAIN = 'data/01_processed/val_sets/cpg_pos_v2.memmap/train'
NEG_TRAIN = 'data/01_processed/val_sets/cpg_neg_v2.memmap/train'
LIMIT = 2_000_000
N_UMAP = 200_000
N_PLOT = 50_000
BATCH_SIZE = 4096


def load_encoder(checkpoint_path, device):
    """Load pretrained SmrtEncoder from SSL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    c = config['smrt2vec']

    encoder = SmrtEncoder(
        d_model=c['d_model'],
        n_layers=c['n_layers'],
        n_head=c['n_head'],
        max_len=c['context'],
    )

    missing, unexpected = encoder.load_state_dict(
        checkpoint['encoder_state_dict'], strict=False
    )
    print(f"Loaded encoder from {checkpoint_path}")
    if missing:
        print(f"  Missing keys (expected for PE resize): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_representations(encoder, dataloader, device):
    """Run encoder inference, return center-position representations."""
    all_reps, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        c = encoder(x)  # [B, T_down, d_model]
        center = c[:, c.shape[1] // 2, :]  # [B, d_model]
        all_reps.append(center.cpu())
        all_labels.append(y)
    return torch.cat(all_reps).numpy(), torch.cat(all_labels).numpy()


def main(output_path):
    alt.data_transformers.enable('vegafusion')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load encoder
    encoder = load_encoder(CHECKPOINT, device)

    # Load CpG dataset with normalization
    tmp_ds = LabeledMemmapDataset(POS_TRAIN, NEG_TRAIN, limit=LIMIT)
    norm = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    ds = LabeledMemmapDataset(POS_TRAIN, NEG_TRAIN, limit=LIMIT, norm_fn=norm, balance=True)
    print(f"Dataset: {len(ds)} samples ({ds.pos_len} pos, {ds.neg_len} neg)")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Extract representations
    print("Running encoder inference...")
    reps, labels = extract_representations(encoder, dl, device)
    print(f"Representations: {reps.shape}, class balance: {labels.mean():.3f}")

    # Subsample for UMAP
    rng = np.random.default_rng(42)
    n_umap = min(N_UMAP, len(reps))
    idx = rng.choice(len(reps), n_umap, replace=False)
    reps_sub = reps[idx]
    labels_sub = labels[idx]

    print(f"Fitting UMAP on {n_umap:,} representations...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1)
    embedding = reducer.fit_transform(reps_sub)
    print("UMAP complete.")

    # Subsample for plotting
    n_plot = min(N_PLOT, len(embedding))
    plot_idx = rng.choice(len(embedding), n_plot, replace=False)

    df = pl.DataFrame({
        'UMAP1': embedding[plot_idx, 0],
        'UMAP2': embedding[plot_idx, 1],
        'class': ['methylated' if yi == 1 else 'unmethylated' for yi in labels_sub[plot_idx]],
    })

    chart = alt.Chart(df).mark_circle(opacity=0.15, size=5).encode(
        alt.X('UMAP1:Q'),
        alt.Y('UMAP2:Q'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8'],
        )),
    ).properties(
        width=600,
        height=600,
        title=f'UMAP of SSL encoder representations — {n_umap:,} CpG samples (d_model=128)',
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
