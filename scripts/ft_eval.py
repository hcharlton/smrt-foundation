"""FT evaluation pipeline for the four-way CpG partition.

Two subcommands, each scoped to one logical unit of work:

    python -m scripts.ft_eval reprobe   --exp_dir <ssl_size_dir>
    python -m scripts.ft_eval evaluate  --ft_exp_dir <ft_dir> [--init_name ssl_58_best]

`reprobe` post-hoc re-runs the linear probe on every saved SSL checkpoint in
one experiment dir against val1 and writes `probe_history_val1.csv` into the
exp_dir. The auto-prefer logic in `scripts.utils.select_best_ssl_checkpoint`
picks it up downstream so `ds_grid_v3.py:421`'s `auto_best` resolution uses the
val1 winner.

`evaluate` walks one FT experiment dir
(`<ft_exp_dir>/<recipe>/<init>/<arch>/n<size>/results.csv`), selects the recipe
with max val_accuracy per (init, arch, size), loads the corresponding
`best_ckpt.pt`, runs a single forward pass over val3, and emits
`inter_ssl_eval.csv` with test_top1/auroc/auprc/f1/loss per artifact.

Orchestration (submit reprobe → wait → submit FT → wait → submit evaluate, with
Slurm `--dependency=afterok` chaining) is deliberately NOT in this file — it's
shell concern, see `docs/methodology.md` for a paste-friendly runbook.

Data paths default to the canonical `cpg_*_v2.memmap/{train,val1,val3}` layout
(see DEFAULT_* constants below) — override only when probing/evaluating against
a non-default partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import SmrtEncoder, SmrtEncoderSmallRF
from smrt_foundation.normalization import KineticsNorm
from scripts.ds_grid_v3 import _build_model

DEFAULT_POS_TRAIN = 'data/01_processed/val_sets/cpg_pos_v2.memmap/train'
DEFAULT_NEG_TRAIN = 'data/01_processed/val_sets/cpg_neg_v2.memmap/train'
DEFAULT_POS_VAL1 = 'data/01_processed/val_sets/cpg_pos_v2.memmap/val1'
DEFAULT_NEG_VAL1 = 'data/01_processed/val_sets/cpg_neg_v2.memmap/val1'
DEFAULT_POS_VAL3 = 'data/01_processed/val_sets/cpg_pos_v2.memmap/val3'
DEFAULT_NEG_VAL3 = 'data/01_processed/val_sets/cpg_neg_v2.memmap/val3'

DEFAULT_RECIPES = 'midlayer,lpft_lldr,decoder_init,recipe_match'


# ---------- Shared helpers --------------------------------------------------

def _step_num(p: Path) -> int:
    try:
        return int(p.stem.split('_')[1])
    except (IndexError, ValueError):
        return -1


def _resolve_cnn_variant(config: dict) -> str:
    """ssl_5[6-9] / ssl_60 default to small_rf; honor explicit overrides."""
    if 'cnn_variant' in config:
        return config['cnn_variant']
    return config.get('smrt2vec', {}).get('cnn_variant', 'small_rf')


def _build_encoder(ssl_cfg: dict, cnn_variant: str) -> nn.Module:
    common = dict(
        d_model=ssl_cfg['d_model'], n_layers=ssl_cfg['n_layers'],
        n_head=ssl_cfg['n_head'], max_len=ssl_cfg['context'],
    )
    if cnn_variant == 'small_rf':
        return SmrtEncoderSmallRF(**common)
    return SmrtEncoder(**common)


def _load_norm(ckpt: dict) -> KineticsNorm:
    """Restore KineticsNorm; tolerate older unprefixed key formats."""
    try:
        return KineticsNorm.load_stats(ckpt)
    except AssertionError:
        # Pre-2026-04 ssl_58 format used unprefixed 'means' / 'stds'.
        return KineticsNorm.from_stats(ckpt['means'], ckpt['stds'])


def _load_encoder_from_ckpt(ckpt_path: Path) -> tuple[nn.Module, KineticsNorm]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ssl_cfg = ckpt['config']['smrt2vec']
    cnn_variant = _resolve_cnn_variant(ckpt['config'])
    enc = _build_encoder(ssl_cfg, cnn_variant)
    enc_sd = {k: v for k, v in ckpt['encoder_state_dict'].items()
              if not (k == 'pe.pe' and v.shape != enc.pe.pe.shape)}
    enc.load_state_dict(enc_sd, strict=False)
    return enc, _load_norm(ckpt)


def _train_probe(
    encoder: nn.Module,
    train_dl: DataLoader, val_dl: DataLoader,
    device: torch.device,
    epochs: int, lr: float,
) -> tuple[float, float]:
    """Freeze encoder, train a fresh Linear(d, 1) head, return (top1, auroc)."""
    encoder.eval()
    head = nn.Linear(encoder.d_model, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        head.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                c = encoder(x)
                centre = c[:, c.shape[1] // 2, :]
            logits = head(centre).squeeze(-1)
            opt.zero_grad()
            crit(logits, y).backward()
            opt.step()

    head.eval()
    acc = BinaryAccuracy().to(device)
    auroc = BinaryAUROC().to(device)
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            c = encoder(x)
            centre = c[:, c.shape[1] // 2, :]
            logits = head(centre).squeeze(-1)
        acc.update(logits > 0, y.long())
        auroc.update(logits, y.long())
    return acc.compute().item(), auroc.compute().item()


def _evaluate_model(model: nn.Module, dl: DataLoader, device: torch.device) -> dict:
    """Single forward pass, all classification metrics + loss."""
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    acc = BinaryAccuracy().to(device)
    auroc = BinaryAUROC().to(device)
    ap = BinaryAveragePrecision().to(device)
    f1 = BinaryF1Score().to(device)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            total_loss += crit(logits, y).item()
            n_batches += 1
            preds = (logits > 0).long()
            y_long = y.long()
            acc.update(preds, y_long)
            auroc.update(logits, y_long)
            ap.update(logits, y_long)
            f1.update(preds, y_long)
    return {
        'test_top1': acc.compute().item(),
        'test_auroc': auroc.compute().item(),
        'test_auprc': ap.compute().item(),
        'test_f1': f1.compute().item(),
        'test_loss': total_loss / max(n_batches, 1),
    }


def _existing_steps(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for r in csv.DictReader(f):
            try:
                done.add(int(r['step']))
            except (KeyError, ValueError):
                continue
    return done


# ---------- Subcommand: reprobe --------------------------------------------

def cmd_reprobe(args) -> None:
    """Re-probe every step_*.pt in args.exp_dir against val1."""
    exp_dir = Path(args.exp_dir).resolve()
    ckpt_dir = exp_dir / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('step_*.pt'), key=_step_num)
    if not ckpts:
        sys.exit(f"No step_*.pt under {ckpt_dir}")

    out_path = exp_dir / args.out_csv
    done = _existing_steps(out_path)
    todo = [p for p in ckpts if _step_num(p) not in done]
    print(f"exp_dir:   {exp_dir}")
    print(f"probe fit: pos={args.probe_pos_train}  neg={args.probe_neg_train}")
    print(f"probe val: pos={args.probe_pos_val}  neg={args.probe_neg_val}")
    print(f"checkpoints: {len(ckpts)} total, {len(done)} already in {out_path.name}, "
          f"{len(todo)} to process")
    if not todo:
        return

    write_header = not out_path.exists()
    out_file = open(out_path, 'a', newline='')
    writer = csv.writer(out_file)
    if write_header:
        writer.writerow(['step', 'probe_top1', 'probe_auroc'])
        out_file.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: no CUDA visible; this will be slow")

    for ckpt_path in todo:
        step = _step_num(ckpt_path)
        print(f"\n[step {step}] {ckpt_path.name}")
        encoder, norm = _load_encoder_from_ckpt(ckpt_path)
        encoder = encoder.to(device)

        train_ds = LabeledMemmapDataset(
            args.probe_pos_train, args.probe_neg_train,
            limit=args.probe_ds_limit, norm_fn=norm, balance=True,
        )
        val_ds = LabeledMemmapDataset(
            args.probe_pos_val, args.probe_neg_val,
            limit=args.probe_ds_limit, norm_fn=norm, balance=True,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

        top1, auroc = _train_probe(
            encoder, train_dl, val_dl, device,
            epochs=args.probe_epochs, lr=args.probe_lr,
        )
        print(f"  top1={top1:.4f} auroc={auroc:.4f}")
        writer.writerow([step, f'{top1:.6f}', f'{auroc:.6f}'])
        out_file.flush()

        del encoder, train_ds, val_ds, train_dl, val_dl
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_file.close()
    print(f"\nWrote {out_path}")


# ---------- Subcommand: evaluate (B + C combined) ---------------------------

def _collect_results(ft_exp_dir: Path, recipes: list[str]) -> pl.DataFrame:
    """Walk every recipe/init/arch/n<size>/results.csv into a single DataFrame.

    Path shape: ft_exp_dir / recipe / init_name / arch / n<size> / results.csv
    """
    frames = []
    for recipe in recipes:
        recipe_dir = ft_exp_dir / recipe
        if not recipe_dir.is_dir():
            print(f"  skip {recipe}: not found", file=sys.stderr)
            continue
        for csv_path in recipe_dir.rglob('results.csv'):
            rel = csv_path.relative_to(recipe_dir)
            if len(rel.parts) != 4:
                continue
            init_name, arch, n_size, _ = rel.parts
            if not n_size.startswith('n'):
                continue
            try:
                size = int(n_size[1:])
            except ValueError:
                continue
            df = pl.read_csv(csv_path).with_columns([
                pl.lit(recipe).alias('recipe'),
                pl.lit(init_name).alias('init_name'),
                pl.lit(arch).alias('arch'),
                pl.lit(size).alias('size'),
                pl.lit(str(csv_path.parent / 'best_ckpt.pt')).alias('ckpt_path'),
            ])
            frames.append(df)
    if not frames:
        sys.exit(f"No results.csv found under {ft_exp_dir} for recipes {recipes}")
    return pl.concat(frames, how='diagonal_relaxed')


def cmd_evaluate(args) -> None:
    """Pick best recipe per (init, arch, size) on val_metric, then eval each on val3."""
    ft_exp_dir = Path(args.ft_exp_dir).resolve()
    recipes = [r.strip() for r in args.recipes.split(',') if r.strip()]

    all_df = _collect_results(ft_exp_dir, recipes)

    if args.init_name is not None:
        all_df = all_df.filter(pl.col('init_name') == args.init_name)
        if all_df.is_empty():
            sys.exit(f"No rows for init_name={args.init_name}")

    if args.metric not in all_df.columns:
        sys.exit(f"Metric column '{args.metric}' not in results.csv columns: {all_df.columns}")

    best = (
        all_df.sort(by=args.metric, descending=True, nulls_last=True)
        .group_by(['init_name', 'arch', 'size'], maintain_order=True)
        .first()
        .select([
            'init_name', 'arch', 'size', 'recipe', 'step',
            pl.col(args.metric).alias('val_metric'),
            'ckpt_path',
        ])
        .sort(['init_name', 'arch', 'size'])
    )
    entries = best.to_dicts()
    print(f"\nSelected {len(entries)} winning recipes:")
    print(best)

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(entries, f, indent=2)
        print(f"Manifest written: {manifest_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: no CUDA visible; eval will be slow")

    print(f"\nEvaluating on val3: pos={args.pos_data_test}  neg={args.neg_data_test}  limit={args.limit:,}")

    rows = []
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] init={entry['init_name']} arch={entry['arch']} "
              f"n={entry['size']} recipe={entry['recipe']}")
        ckpt_path = Path(entry['ckpt_path'])
        if not ckpt_path.exists():
            print(f"  SKIP {ckpt_path}: file not found", file=sys.stderr)
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch_cfg = ckpt['arch_cfg']
        model = _build_model(arch_cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        norm = KineticsNorm.load_stats(ckpt)

        ds = LabeledMemmapDataset(
            args.pos_data_test, args.neg_data_test,
            limit=args.limit, norm_fn=norm, balance=True,
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
        metrics = _evaluate_model(model, dl, device)
        for k, v in metrics.items():
            print(f"  {k}={v:.4f}")
        rows.append({**entry, **metrics})

        del model, ds, dl
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not rows:
        sys.exit("No usable entries; nothing written")

    df = pl.DataFrame(rows).sort('test_top1', descending=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(df.head(5))

    best_row = df.row(0, named=True)
    print("\n=== Best Finetuned Artifact ===")
    print(f"  init={best_row['init_name']}  arch={best_row['arch']}  "
          f"size={best_row['size']}  recipe={best_row['recipe']}")
    print(f"  test_top1={best_row['test_top1']:.4f}  test_auroc={best_row['test_auroc']:.4f}  "
          f"test_auprc={best_row['test_auprc']:.4f}  test_f1={best_row['test_f1']:.4f}")
    print(f"  ckpt={best_row['ckpt_path']}")


# ---------- CLI -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FT evaluation pipeline")
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('reprobe', help='Re-probe every SSL checkpoint on val1')
    r.add_argument('--exp_dir', required=True,
                   help='SSL experiment dir with config.yaml and checkpoints/step_*.pt')
    r.add_argument('--probe_pos_train', default=DEFAULT_POS_TRAIN)
    r.add_argument('--probe_neg_train', default=DEFAULT_NEG_TRAIN)
    r.add_argument('--probe_pos_val', default=DEFAULT_POS_VAL1,
                   help='Default points at val1 — the canonical SSL-selection partition.')
    r.add_argument('--probe_neg_val', default=DEFAULT_NEG_VAL1)
    r.add_argument('--probe_epochs', type=int, default=3)
    r.add_argument('--probe_lr', type=float, default=3e-3)
    r.add_argument('--probe_ds_limit', type=int, default=500_000)
    r.add_argument('--batch_size', type=int, default=512)
    r.add_argument('--num_workers', type=int, default=2)
    r.add_argument('--out_csv', default='probe_history_val1.csv',
                   help='Filename written into <exp_dir>.')
    r.set_defaults(func=cmd_reprobe)

    e = sub.add_parser('evaluate',
                       help='Pick best recipe per (init, arch, size); eval winners on val3.')
    e.add_argument('--ft_exp_dir', required=True,
                   help='FT experiment root (parent of recipe subdirs)')
    e.add_argument('--init_name', default=None,
                   help='Optional init filter (e.g. ssl_58_best)')
    e.add_argument('--metric', default='val_accuracy')
    e.add_argument('--recipes', default=DEFAULT_RECIPES)
    e.add_argument('--pos_data_test', default=DEFAULT_POS_VAL3)
    e.add_argument('--neg_data_test', default=DEFAULT_NEG_VAL3)
    e.add_argument('--limit', type=int, default=1_000_000,
                   help='Per-class cap on val3 samples')
    e.add_argument('--batch_size', type=int, default=512)
    e.add_argument('--num_workers', type=int, default=2)
    e.add_argument('--manifest_out', default=None,
                   help='Optional: also write the JSON manifest of winners (debug)')
    e.add_argument('--out', default='inter_ssl_eval.csv')
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
