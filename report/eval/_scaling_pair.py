"""
Shared driver for the data-scaling trio: scaling curve, residuals, and
per-train_size trajectory facets. Given two CSVs (a supervised baseline
and a finetuned run), produces all three plots into an output directory.

Plot directories in report/eval/ use this module via a thin plot.py stub
and a minimal config.yaml — see report/eval/data_scaling_pair/ for the
reference layout.
"""

import os
import yaml
import polars as pl
import altair as alt


RESIDUALS_SUFFIX = '_residuals'
TRAJECTORIES_SUFFIX = '_trajectories'


def load_rows(path, label):
    return (
        pl.read_csv(path)
        .select('train_size', 'step', 'val_accuracy')
        .with_columns(pl.lit(label).alias('model'))
    )


def best_per_train_size(df):
    return (
        df.sort('val_accuracy', descending=True)
        .group_by('train_size', 'model')
        .first()
        .select('train_size', 'val_accuracy', 'model')
    )


def format_size(n):
    if n >= 1_000_000:
        return f'{n // 1_000_000}M'
    if n >= 1_000:
        return f'{n // 1_000}K'
    return str(n)


def build_scaling_chart(df_best, cfg):
    return alt.Chart(df_best.sort('model', 'train_size')).mark_line().encode(
        alt.X('train_size:Q', scale=alt.Scale(type='log', base=2)).title(cfg.get('x_label', 'x')),
        alt.Y('val_accuracy:Q', scale=alt.Scale(zero=False)).title(cfg.get('y_label', 'y')),
        alt.Color('model:N').title('Model'),
    ).properties(
        width=cfg.get('width', 500),
        height=cfg.get('height', 300),
        title=cfg.get('title', ''),
    )


def build_residuals_chart(df_best, baseline_label, cfg):
    baseline = (
        df_best.filter(pl.col('model') == baseline_label)
        .select('train_size', pl.col('val_accuracy').alias('baseline_accuracy'))
    )
    df = (
        df_best.join(baseline, on='train_size', how='inner')
        .with_columns((pl.col('val_accuracy') - pl.col('baseline_accuracy')).alias('residual'))
        .sort('model', 'train_size')
    )

    train_sizes = sorted(df['train_size'].unique().to_list())
    return alt.Chart(df).mark_line().encode(
        alt.X(
            'train_size:Q',
            scale=alt.Scale(type='log', base=2),
            axis=alt.Axis(values=train_sizes),
        ).title(cfg.get('x_label', 'x')),
        alt.Y('residual:Q', scale=alt.Scale(zero=True)).title(cfg.get('residual_y_label', 'y')),
        alt.Color('model:N').title('Model'),
    ).properties(
        width=cfg.get('width', 500),
        height=cfg.get('height', 300),
        title=cfg.get('title', ''),
    )


def build_trajectories_chart(df_all, cfg):
    sizes_sorted = sorted(df_all['train_size'].unique().to_list())
    size_to_label = {s: format_size(s) for s in sizes_sorted}
    labels_sorted = [size_to_label[s] for s in sizes_sorted]

    df = df_all.with_columns(
        pl.col('train_size')
        .replace_strict(size_to_label, return_dtype=pl.String)
        .alias('train_size_label')
    ).sort('model', 'train_size', 'step')

    return alt.Chart(df).mark_line().encode(
        alt.X('step:Q').title(cfg.get('trajectories_x_label', 'Step')),
        alt.Y('val_accuracy:Q', scale=alt.Scale(zero=False)).title(
            cfg.get('trajectories_y_label', 'Top-1 Accuracy')
        ),
        alt.Color('model:N').title('Model'),
    ).properties(
        width=cfg.get('facet_width', 200),
        height=cfg.get('facet_height', 150),
    ).facet(
        facet=alt.Facet(
            'train_size_label:N',
            sort=labels_sorted,
            title='Training dataset size',
        ),
        columns=cfg.get('columns', 4),
    ).properties(
        title=cfg.get('title', ''),
    )


def _sibling_path(output_path, suffix):
    stem, ext = os.path.splitext(output_path)
    return f'{stem}{suffix}{ext}'


def main(config_path, output_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    baseline_cfg = cfg['baseline']
    finetuned_cfg = cfg['finetuned']

    df_all = pl.concat([
        load_rows(baseline_cfg['path'], baseline_cfg['label']),
        load_rows(finetuned_cfg['path'], finetuned_cfg['label']),
    ])
    df_best = best_per_train_size(df_all)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    charts = [
        (output_path, build_scaling_chart(df_best, cfg)),
        (_sibling_path(output_path, RESIDUALS_SUFFIX), build_residuals_chart(df_best, baseline_cfg['label'], cfg)),
        (_sibling_path(output_path, TRAJECTORIES_SUFFIX), build_trajectories_chart(df_all, cfg)),
    ]
    for path, chart in charts:
        chart.save(path)
        print(f'Saved to {path}')
