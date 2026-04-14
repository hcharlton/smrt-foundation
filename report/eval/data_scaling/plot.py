"""
Compares directly trained model against finetuned 
pretrained encoder at different training dataset sizes. 
The validation dataset is the same across each run. 
"""

import os
import sys
import yaml
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def main(output_path):
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    df = pl.read_csv(config['data_path']).sort('val_accuracy', descending=True).group_by('train_size').first()
    # print(df.head())
    # --- Build chart ---
    chart = alt.Chart(df).mark_line(point=True).encode(
        alt.X('train_size:Q',scale=alt.Scale(type='log', base=2)).title(config.get('x_label', 'x')),
        alt.Y('val_accuracy:Q',scale=alt.Scale(zero=False)).title(config.get('y_label', 'y')),
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
