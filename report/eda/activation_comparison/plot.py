"""
Activation comparison: legacy vs new dataset pipeline.

Loads the .npz saved by supervised_19_activation_cmp/train.py and
generates diagnostic plots comparing encoder representations.

Panels:
  1. PCA of center activations colored by class, faceted by pipeline
  2. Logit distributions by class and pipeline
  3. Per-dimension mean activation difference (pos - neg) for each pipeline
"""

import os
import sys
import argparse
import numpy as np
import polars as pl
import altair as alt
from sklearn.decomposition import PCA

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


NPZ_PATH = 'scripts/experiments/supervised_19_activation_cmp/activations.npz'


def main(output_path):
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} not found. Run the training experiment first.")
        sys.exit(1)

    data = np.load(NPZ_PATH)
    leg_acts = data['legacy_activations']
    leg_logits = data['legacy_logits']
    leg_labels = data['legacy_labels']
    new_acts = data['new_activations']
    new_logits = data['new_logits']
    new_labels = data['new_labels']

    # Subsample for plotting speed
    rng = np.random.default_rng(42)
    max_plot = 5000
    if len(leg_acts) > max_plot:
        idx = rng.choice(len(leg_acts), max_plot, replace=False)
        leg_acts, leg_logits, leg_labels = leg_acts[idx], leg_logits[idx], leg_labels[idx]
    if len(new_acts) > max_plot:
        idx = rng.choice(len(new_acts), max_plot, replace=False)
        new_acts, new_logits, new_labels = new_acts[idx], new_logits[idx], new_labels[idx]

    # ---- Panel 1: PCA of center activations ----
    combined_acts = np.concatenate([leg_acts, new_acts], axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(combined_acts)

    n_leg = len(leg_acts)
    pca_df = pl.DataFrame({
        'PC1': pca_coords[:, 0].astype(np.float64),
        'PC2': pca_coords[:, 1].astype(np.float64),
        'class': ['pos' if l > 0.5 else 'neg' for l in np.concatenate([leg_labels, new_labels])],
        'pipeline': ['legacy'] * n_leg + ['new'] * len(new_acts),
    })

    pca_chart = alt.Chart(pca_df).mark_circle(size=8, opacity=0.4).encode(
        alt.X('PC1:Q').title(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)'),
        alt.Y('PC2:Q').title(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['pos', 'neg'],
            range=['#e45756', '#4c78a8']
        )),
        alt.Column('pipeline:N'),
    ).properties(width=300, height=300, title='PCA of encoder center activations')

    # ---- Panel 2: Logit distributions ----
    logit_df = pl.DataFrame({
        'logit': np.concatenate([leg_logits, new_logits]).astype(np.float64),
        'class': ['pos' if l > 0.5 else 'neg' for l in np.concatenate([leg_labels, new_labels])],
        'pipeline': ['legacy'] * n_leg + ['new'] * len(new_logits),
    })

    logit_chart = alt.Chart(logit_df).mark_area(
        opacity=0.5, interpolate='step'
    ).encode(
        alt.X('logit:Q').bin(maxbins=50).title('Logit (pre-sigmoid)'),
        alt.Y('count():Q').stack(None).title('Count'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['pos', 'neg'],
            range=['#e45756', '#4c78a8']
        )),
        alt.Column('pipeline:N'),
    ).properties(width=300, height=200, title='Logit distributions by class')

    # ---- Panel 3: Per-dimension discriminability ----
    def dim_diff(acts, labels):
        pos_mask = labels > 0.5
        pos_mean = acts[pos_mask].mean(axis=0)
        neg_mean = acts[~pos_mask].mean(axis=0)
        return pos_mean - neg_mean

    leg_diff = dim_diff(leg_acts, leg_labels)
    new_diff = dim_diff(new_acts, new_labels)

    dim_df = pl.DataFrame({
        'dimension': list(range(len(leg_diff))) + list(range(len(new_diff))),
        'pos_minus_neg_mean': np.concatenate([leg_diff, new_diff]).astype(np.float64),
        'pipeline': ['legacy'] * len(leg_diff) + ['new'] * len(new_diff),
    })

    dim_chart = alt.Chart(dim_df).mark_bar(opacity=0.7).encode(
        alt.X('dimension:O').title('Activation dimension'),
        alt.Y('pos_minus_neg_mean:Q').title('Mean(pos) - Mean(neg)'),
        alt.Color('pipeline:N', scale=alt.Scale(
            domain=['legacy', 'new'],
            range=['#e45756', '#4c78a8']
        )),
        alt.Column('pipeline:N'),
    ).properties(width=500, height=200, title='Per-dimension class discriminability')

    # ---- Combine ----
    final = alt.vconcat(pca_chart, logit_chart, dim_chart).resolve_scale(
        color='independent'
    )

    final.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
