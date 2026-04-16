"""
Plot residuals of best top-1 accuracy against a baseline series, as a
function of training dataset size. For each (series, train_size) take
the best val_accuracy, then subtract the baseline's best at the same
train_size. Only train_sizes present in the baseline are included.
"""

import os
import sys
import yaml
import argparse
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def load_best(path, label):
    return (
        pl.read_csv(path)
        .sort('val_accuracy', descending=True)
        .group_by('train_size')
        .first()
        .select('train_size', 'val_accuracy')
        .with_columns(pl.lit(label).alias('model'))
    )


def main(output_path):
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df_all = pl.concat([
        load_best(s['path'], s['label']) for s in config['series']
    ])

    baseline = (
        df_all.filter(pl.col('model') == config['baseline'])
        .select('train_size', pl.col('val_accuracy').alias('baseline_accuracy'))
    )

    df = (
        df_all.join(baseline, on='train_size', how='inner')
        .with_columns((pl.col('val_accuracy') - pl.col('baseline_accuracy')).alias('residual'))
        .sort('model', 'train_size')
    )

    chart = alt.Chart(df).mark_line().encode(
        alt.X('train_size:Q', scale=alt.Scale(type='log', base=2)).title(config.get('x_label', 'x')),
        alt.Y('residual:Q', scale=alt.Scale(zero=True)).title(config.get('y_label', 'y')),
        alt.Color('model:N').title('Model'),
    ).properties(
        width=config.get('width', 500),
        height=config.get('height', 300),
        title=config.get('title', ''),
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
