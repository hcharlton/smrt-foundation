import polars as pl
import os
import yaml
import argparse
from smrt_foundation.utils import parse_yaml
from smrt_foundation.dataset import SCHEMA, PER_BASE_TAGS, REQUIRED_TAGS

def compute_log_normalization_stats(df, features, epsilon=1):
    means = {col: (df[col].explode() + epsilon).log().mean() for col in features}
    stds = {col: (df[col].explode() + epsilon).log().explode().std() for col in features}
    output_dict = {'log_norm':{
        'means': means,
        'stds': stds
    }}
    return output_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generates normalization statistics based on the train partition of methylation data"
    )
    parser.add_argument('-i', '--input_path',
                        type = str,
                        required=True,
                        help="Input filepath to the training data")
    parser.add_argument('-o', '--output_path',
                        type=str,
                        required=True,
                        help="Path to output file (including filename)")
    parser.add_argument('-c', '--config_path',
                        type=str,
                        help='path to config file')
    parser.add_argument('-t', '--truncate',
                        action='store_true',
                        help='If specified, only uses first 10_000 samples')
    
    args = parser.parse_args()
    print('began running compute stats')
    config = parse_yaml(args.config_path)
    q = (
        pl.scan_parquet(os.path.expanduser(args.input_path),
                        schema = config['data']['schema']
                        )
                    )

    df = q.collect()
    print('collected df')
    exp_outpath = args.output_path

    kinetics_features = config['data']['kinetics_features']

    stats_dict = compute_log_normalization_stats(df, kinetics_features)
    print('computed stats')
    os.makedirs(os.path.dirname(exp_outpath), exist_ok=True)

    with open(exp_outpath, 'w') as f:
        yaml.dump(stats_dict, f, indent=4)

    print(f"Normalization stats saved to {exp_outpath}")
    print(yaml.dump(stats_dict, indent=4))


if __name__ == "__main__":
    main()