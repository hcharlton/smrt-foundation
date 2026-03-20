"""
Raw feature comparison: new memmap pipeline vs legacy parquet pipeline.

Overlaid histograms comparing the raw (pre-normalization) values of each
feature (sequence tokens, IPD, pulse width) as produced by each pipeline.
Each pane contains two series: 'new' and 'legacy'.
"""

import os
import sys
import yaml
import glob
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def load_all_shards(directory, max_samples=50_000):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    arrays = []
    n = 0
    for p in paths:
        arr = np.load(p)
        arrays.append(arr)
        n += arr.shape[0]
        if max_samples and n >= max_samples:
            break
    out = np.concatenate(arrays, axis=0)
    return out[:max_samples] if max_samples else out


def main(output_path):
    with open('./configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    # Fallback to subset data if full dataset not available
    if not os.path.isdir(os.path.expandvars(sup_config.get('pos_data_train', ''))):
        sup_config['pos_data_train'] = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
        sup_config['neg_data_train'] = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'

    context = data_config['cpg_pipeline']['context']

    # ---- New pipeline: raw memmap ----
    pos_new = load_all_shards(sup_config['pos_data_train'])
    neg_new = load_all_shards(sup_config['neg_data_train'])
    all_new = np.concatenate([pos_new, neg_new], axis=0)
    mask_new = all_new[:, :, -1] == 0.0

    rng = np.random.default_rng(42)
    n = min(50_000, mask_new.sum())

    new_seq = all_new[:, :, 0][mask_new].flatten().astype(np.float64)
    new_ipd = all_new[:, :, 1][mask_new].flatten().astype(np.float64)
    new_pw = all_new[:, :, 2][mask_new].flatten().astype(np.float64)

    idx = rng.choice(len(new_seq), n, replace=False)
    new_seq, new_ipd, new_pw = new_seq[idx], new_ipd[idx], new_pw[idx]

    # ---- Legacy pipeline: raw parquet ----
    legacy_path = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
    if not os.path.exists(legacy_path):
        legacy_path = 'data/01_processed/val_sets/legacy_subset_train.parquet'

    df = pl.read_parquet(legacy_path)
    if len(df) > 50_000:
        df = df.sample(n=50_000, seed=42)

    kin_feats = ['fi', 'fp']
    df = df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])

    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    legacy_seq_tokens = []
    for s in df['seq'].to_list():
        legacy_seq_tokens.extend(vocab.get(c, 4) for c in s)
    legacy_seq = np.array(legacy_seq_tokens, dtype=np.float64)

    legacy_ipd = df['fi'].to_numpy().astype(np.float64).flatten()
    legacy_pw = df['fp'].to_numpy().astype(np.float64).flatten()

    # Subsample legacy to same size
    n_leg = min(n, len(legacy_seq))
    idx_l = rng.choice(len(legacy_seq), n_leg, replace=False)
    legacy_seq, legacy_ipd, legacy_pw = legacy_seq[idx_l], legacy_ipd[idx_l], legacy_pw[idx_l]

    # ---- Build charts ----
    features = [
        ('Sequence token', new_seq, legacy_seq),
        ('IPD (fi)', new_ipd, legacy_ipd),
        ('Pulse width (fp)', new_pw, legacy_pw),
    ]

    charts = []
    for feat_name, new_vals, leg_vals in features:
        n_min = min(len(new_vals), len(leg_vals))
        feat_df = pl.DataFrame({
            'value': np.concatenate([new_vals[:n_min], leg_vals[:n_min]]),
            'pipeline': ['new'] * n_min + ['legacy'] * n_min,
        })

        chart = alt.Chart(feat_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title(feat_name),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('pipeline:N', scale=alt.Scale(
                domain=['new', 'legacy'],
                range=['#4c78a8', '#e45756']
            )),
        ).properties(width=500, height=200, title=feat_name)
        charts.append(chart)

    final = alt.vconcat(*charts).properties(
        title='Raw feature comparison: new pipeline vs legacy'
    )

    final.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
