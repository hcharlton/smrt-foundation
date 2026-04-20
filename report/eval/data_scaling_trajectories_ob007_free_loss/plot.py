"""
Facet plot of train and validation loss trajectories vs step for
experiments 33, 37, 38. One facet per training dataset size; two
lines per experiment (train_loss solid, val_loss dashed) coloured by
model. Independent x and y scales per facet so each trajectory uses
the full plotting area.
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


def format_size(n):
    if n >= 1_000_000:
        return f'{n // 1_000_000}M'
    if n >= 1_000:
        return f'{n // 1_000}K'
    return str(n)


def load_all(path, label):
    return (
        pl.read_csv(path)
        .select('train_size', 'step', 'train_loss', 'val_loss')
        .with_columns(pl.lit(label).alias('model'))
    )


def main(output_path):
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = pl.concat([
        load_all(s['path'], s['label']) for s in config['series']
    ])

    df = df.unpivot(
        index=['train_size', 'step', 'model'],
        on=['train_loss', 'val_loss'],
        variable_name='loss_type',
        value_name='loss',
    ).with_columns(
        pl.col('loss_type').replace_strict(
            {'train_loss': 'train', 'val_loss': 'val'},
            return_dtype=pl.String,
        )
    ).drop_nulls('loss')

    sizes_sorted = sorted(df['train_size'].unique().to_list())
    size_to_label = {s: format_size(s) for s in sizes_sorted}
    labels_sorted = [size_to_label[s] for s in sizes_sorted]

    df = df.with_columns(
        pl.col('train_size')
        .replace_strict(size_to_label, return_dtype=pl.String)
        .alias('train_size_label')
    ).sort('model', 'train_size', 'loss_type', 'step')

    chart = alt.Chart(df).mark_line().encode(
        alt.X('step:Q').title(config.get('x_label', 'Step')),
        alt.Y('loss:Q', scale=alt.Scale(zero=False)).title(config.get('y_label', 'Loss')),
        alt.Color('model:N').title('Model'),
        alt.StrokeDash('loss_type:N', sort=['train', 'val']).title('Split'),
    ).properties(
        width=config.get('facet_width', 200),
        height=config.get('facet_height', 150),
    ).facet(
        facet=alt.Facet(
            'train_size_label:N',
            sort=labels_sorted,
            title='Training dataset size',
        ),
        columns=config.get('columns', 4),
    ).resolve_scale(
        x='independent',
        y='independent',
    ).properties(
        title=config.get('title', ''),
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
