"""
Facet plot of val_accuracy vs step for the batch-size ablation:
exp 43 (bs=4096) vs exp 44 (bs scaled with train_size). One facet
per training dataset size, independent scales per facet.
"""

import os
import argparse
import yaml
import polars as pl
import altair as alt


def format_size(n):
    if n >= 1_000_000:
        return f'{n // 1_000_000}M'
    if n >= 1_000:
        return f'{n // 1_000}K'
    return str(n)


def main(output_path):
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    cfg = yaml.safe_load(open(cfg_path))

    df = pl.concat([
        pl.read_csv(s['path'])
          .select('train_size', 'step', 'val_accuracy')
          .with_columns(pl.lit(s['label']).alias('model'))
        for s in cfg['series']
    ])

    sizes = sorted(df['train_size'].unique().to_list())
    labels = [format_size(s) for s in sizes]
    df = df.with_columns(
        pl.col('train_size')
          .replace_strict(dict(zip(sizes, labels)), return_dtype=pl.String)
          .alias('size')
    )

    chart = alt.Chart(df).mark_line().encode(
        alt.X('step:Q').title(cfg.get('x_label', 'Step')),
        alt.Y('val_accuracy:Q', scale=alt.Scale(zero=False)).title(cfg.get('y_label', 'Top-1 Accuracy')),
        alt.Color('model:N').title('Model'),
    ).properties(
        width=cfg.get('facet_width', 200),
        height=cfg.get('facet_height', 150),
    ).facet(
        facet=alt.Facet('size:N', sort=labels, title='Training dataset size'),
        columns=cfg.get('columns', 4),
    ).resolve_scale(x='independent', y='independent').properties(title=cfg.get('title', ''))

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output_path', required=True)
    main(p.parse_args().output_path)
