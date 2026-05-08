"""Layer-by-layer linear probe on a frozen SSL encoder.

For each transformer layer index in [0, n_layers - 1], train a fresh
linear probe on labeled CpG data using the encoder's output at that
layer (via SmrtEncoder.forward_to_layer), then write
midlayer_probe_results.csv with one row per (size, layer_idx).

The output identifies the empirical best read-out layer per ssl_58 size,
which then feeds into supervised_53_finetune_revamp/midlayer/config.yaml's
per-arch `layer_idx` knob.

Usage:
    python -m scripts.utils.midlayer_probe_sweep \\
        --ssl_root scripts/experiments/ssl_58_autoencoder_grid \\
        --sizes d128_L4,d256_L8,d512_L8,d768_L8 \\
        --pos_data data/01_processed/val_sets/cpg_pos_v2.memmap/train \\
        --neg_data data/01_processed/val_sets/cpg_neg_v2.memmap/train \\
        --pos_val  data/01_processed/val_sets/cpg_pos_v2.memmap/val \\
        --neg_val  data/01_processed/val_sets/cpg_neg_v2.memmap/val \\
        --out report/midlayer_probe_results.csv

Runs on a single GPU (one size at a time). ~30 min per size at default
settings (probe_limit=200k, 3 epochs per layer).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import SmrtEncoderSmallRF
from smrt_foundation.normalization import KineticsNorm

from scripts.utils.select_best_ssl_checkpoint import select_best_ssl_checkpoint


def _load_encoder_for_size(ssl_root: Path, size_name: str) -> tuple[nn.Module, KineticsNorm]:
    size_dir = ssl_root / f'size_{size_name}'
    ckpt_path = select_best_ssl_checkpoint(size_dir)
    print(f"  Best-probe checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ssl_cfg = ckpt['config']['smrt2vec']
    enc = SmrtEncoderSmallRF(
        d_model=ssl_cfg['d_model'], n_layers=ssl_cfg['n_layers'],
        n_head=ssl_cfg['n_head'], max_len=ssl_cfg['context'],
    )
    enc_sd = ckpt['encoder_state_dict']
    # Tolerate PE shape mismatch (probe context=32, SSL context=512).
    enc_sd = {k: v for k, v in enc_sd.items()
              if not (k == 'pe.pe' and v.shape != enc.pe.pe.shape)}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected encoder keys: {unexpected[:4]}")
    norm = KineticsNorm.load_stats({'means': ckpt['means'], 'stds': ckpt['stds']})
    return enc, norm


def _train_probe_at_layer(
    encoder: nn.Module, layer_idx: int,
    train_dl: DataLoader, val_dl: DataLoader,
    device: torch.device,
    epochs: int = 3, lr: float = 3e-3,
) -> tuple[float, float]:
    """Freeze encoder, train one Linear head at layer_idx, return (top1, auroc)."""
    encoder.eval()
    d = encoder.d_model
    head = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        head.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                c = encoder.forward_to_layer(x, layer_idx)
                centre = c[:, c.shape[1] // 2, :]
            logits = head(centre).squeeze(-1)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    acc = BinaryAccuracy().to(device)
    auroc = BinaryAUROC().to(device)
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            c = encoder.forward_to_layer(x, layer_idx)
            centre = c[:, c.shape[1] // 2, :]
            logits = head(centre).squeeze(-1)
        acc.update(logits > 0, y.long())
        auroc.update(logits, y.long())
    return acc.compute().item(), auroc.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ssl_root', required=True,
                    help='SSL experiment root (e.g. scripts/experiments/ssl_58_autoencoder_grid)')
    ap.add_argument('--sizes', required=True,
                    help='Comma-separated list of size dirs without the size_ prefix')
    ap.add_argument('--pos_data', required=True)
    ap.add_argument('--neg_data', required=True)
    ap.add_argument('--pos_val', required=True)
    ap.add_argument('--neg_val', required=True)
    ap.add_argument('--out', default='report/midlayer_probe_results.csv')
    ap.add_argument('--probe_limit', type=int, default=200_000)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-3)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: no CUDA visible; this will be very slow")

    ssl_root = Path(args.ssl_root).resolve()
    sizes = [s.strip() for s in args.sizes.split(',') if s.strip()]

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(['size', 'layer_idx', 'probe_top1', 'probe_auroc'])

    for size in sizes:
        print(f"\n=== Size: {size} ===")
        try:
            encoder, norm = _load_encoder_for_size(ssl_root, size)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        encoder = encoder.to(device)
        n_layers = len(encoder.blocks)
        print(f"  Encoder: d={encoder.d_model}, n_layers={n_layers}")

        train_ds = LabeledMemmapDataset(
            args.pos_data, args.neg_data,
            limit=args.probe_limit, norm_fn=norm, balance=True,
        )
        val_ds = LabeledMemmapDataset(
            args.pos_val, args.neg_val,
            limit=args.probe_limit, norm_fn=norm, balance=True,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        for layer_idx in range(n_layers):
            top1, auroc = _train_probe_at_layer(
                encoder, layer_idx, train_dl, val_dl, device,
                epochs=args.epochs, lr=args.lr,
            )
            print(f"  layer {layer_idx}: top1={top1:.4f}  auroc={auroc:.4f}")
            with open(out_path, 'a', newline='') as f:
                csv.writer(f).writerow([size, layer_idx, f'{top1:.6f}', f'{auroc:.6f}'])

        del encoder, train_ds, val_ds, train_dl, val_dl
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
