# Linear separability of CpG methylation

Tests whether CpG methylation status is linearly separable from kinetics features using logistic regression.

## Data

- Source: v2 memmap pipeline (`cpg_pos_v2.memmap`, `cpg_neg_v2.memmap`)
- Normalization: `KineticsNorm` (log1p + z-score on kinetics channels, excluding padding)
- Balanced classes, 2M total samples (1M per class)
- Train/val split via existing directory structure

## Features

Each sample is a 32-base window centered on a CpG dinucleotide with 2 kinetics features per position (IPD and pulse width):

- **Full context model**: all 32 positions x 2 features = 64 dimensions
- **Center-6 model**: positions 13-18 (3 on each side of CpG at 15-16) x 2 features = 12 dimensions

## Plots

| File | Description |
|------|-------------|
| `plot.svg` | ROC curves comparing both models |
| `calibration.svg` | Calibration plot (predicted probability vs observed frequency) — shows whether the model's confidence is meaningful |
| `coefficients.svg` | Logistic regression coefficients as a heatmap (32 positions x 2 features) — shows which positions and features are most discriminative |
| `center6_scatter.svg` | PCA projection of center-6 features colored by class |

## Running

```bash
bash plot.sh report/linear              # on cluster
bash plot.sh report/linear --mem=64gb   # if more memory needed
```
