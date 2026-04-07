"""
Empirically measure how many indices end up masked under this experiment's
Smrt2VecInputMask configuration, at both the input resolution (T) and the
latent resolution (T/4, where the AgInfoNCE loss actually applies).

Why: with p_mask sampled at every position and a contiguous-span expansion
of `mask_size`, plus the conservative downsample rule (a latent is masked
if ANY of its 4 input positions were masked), the effective masked fraction
is much larger than `p_mask` and depends nonlinearly on `mask_size` and `T`.
This script makes the actual numbers visible alongside the AgInfoNCE batch
math, so it's clear how many positives the loss is asked to discriminate
per step and how many of them survive `max_negatives` subsampling.

Run locally on CPU:
    python -m scripts.experiments.ssl_26_cpg_contrastive.measure_mask_fractions
"""

import os
import sys

import torch
import yaml

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.model import Smrt2VecInputMask


N_SAMPLES = 200_000  # large enough for tight CIs on per-row statistics
SEED      = 42


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    c = config.get('smrt2vec', {})

    T             = int(c['context'])
    p_mask        = float(c['p_mask'])
    mask_size     = int(c['mask_size'])
    batch_size    = int(c['batch_size'])
    max_negatives = int(c.get('max_negatives', 256))
    n_proc        = int(config.get('resources', {}).get('num_processes', 1))

    torch.manual_seed(SEED)

    # Instantiate the full module so the measurement uses the exact code
    # path that train.py exercises (apply_input_mask + _downsample_mask).
    m = Smrt2VecInputMask(
        d_model=c.get('d_model', 128),
        n_layers=c.get('n_layers', 4),
        n_head=c.get('n_head', 4),
        max_len=T,
        p_mask=p_mask,
        mask_size=mask_size,
    )

    # CpG windows have no padding (always T real bases), so the pad channel
    # is left at zero. The other channels are irrelevant to mask sampling.
    x = torch.zeros(N_SAMPLES, T, 4)

    _, input_mask = m.apply_input_mask(x, p_mask, mask_size)
    ds_mask = m._downsample_mask(input_mask, target_len=T // 4)

    input_per_row  = input_mask.float().mean(dim=1)
    latent_per_row = ds_mask.float().mean(dim=1)

    exp_name = config.get('experiment_name', os.path.basename(os.path.dirname(__file__)))
    print(f"Empirical masking fractions for exp '{exp_name}'")
    print(f"  context (T)             : {T}")
    print(f"  p_mask                  : {p_mask}")
    print(f"  mask_size               : {mask_size}")
    print(f"  samples drawn           : {N_SAMPLES}")
    print()

    print(f"INPUT RESOLUTION  (T = {T})")
    print(f"  overall masked fraction : {input_mask.float().mean().item():.4f}")
    print(f"  per-row mean / std      : "
          f"{input_per_row.mean().item():.4f} / {input_per_row.std().item():.4f}")
    print(f"  per-row min/median/max  : "
          f"{input_per_row.min().item():.4f} / "
          f"{input_per_row.median().item():.4f} / "
          f"{input_per_row.max().item():.4f}")
    print(f"  rows w/ 0 masked        : {(input_per_row == 0).float().mean().item():.4f}")
    print(f"  rows w/ all masked      : {(input_per_row == 1).float().mean().item():.4f}")
    print()

    print(f"LATENT RESOLUTION  (T/4 = {T // 4})")
    print(f"  overall masked fraction : {ds_mask.float().mean().item():.4f}")
    print(f"  per-row mean / std      : "
          f"{latent_per_row.mean().item():.4f} / {latent_per_row.std().item():.4f}")
    print(f"  per-row min/median/max  : "
          f"{latent_per_row.min().item():.4f} / "
          f"{latent_per_row.median().item():.4f} / "
          f"{latent_per_row.max().item():.4f}")
    counts = ds_mask.sum(dim=1)
    for k in range(T // 4 + 1):
        print(f"  rows w/ exactly {k} masked: "
              f"{(counts == k).float().mean().item():.4f}")
    print()

    theo = 1 - (1 - p_mask) ** mask_size
    print(f"THEORETICAL (interior input position, pre-downsample)")
    print(f"  P(masked) = 1 - (1 - {p_mask})^{mask_size} = {theo:.4f}")
    print()

    masked_per_batch = batch_size * latent_per_row.mean().item() * (T // 4)
    print(f"AgInfoNCE BATCH MATH (per GPU)")
    print(f"  bs={batch_size}, T/4={T // 4}, max_negatives={max_negatives}, world_size={n_proc}")
    print(f"  masked latents per local batch  : ~{masked_per_batch:,.0f}")
    print(f"  fraction kept after subsample   : {max_negatives / masked_per_batch:.4%}")
    print(f"  effective negatives after gather: "
          f"{max_negatives * n_proc} = {max_negatives} x {n_proc}")


if __name__ == '__main__':
    main()
