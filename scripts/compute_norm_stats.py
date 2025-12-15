# import polars as pl
# import os
# import yaml
import jax.numpy as jnp
import jax
import tensorstore as ts
import concurrent.futures
import zarr
# import argparse
# from smrt_foundation.utils import parse_yaml


def read_chunk(args):
    zarr_path, chunk_idx,  data_step, chunk_len = args
    z = zarr.open(zarr_path, mode='r')
    start = chunk_idx*chunk_len
    end = min(start+chunk_len, z['data'].shape[0])
    return z['data'][start:end:data_step]

def compute_stats(zarr_path, chunk_step=20, data_step=100, num_threads=14):
    z = zarr.open(zarr_path, mode='r')
    data = z['data']

    total_chunks = data.cdata_shape[0]
    chunk_len = data.chunks[0]

    indices = range(0, total_chunks, chunk_step)
    tasks = [(zarr_path, i, data_step, chunk_len) for i in indices]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        sampled_chunks = list(executor.map(read_chunk, tasks))

    collected_sample = jnp.concatenate(sampled_chunks)

    means = jnp.mean(collected_sample [:,1:],axis=0)
    stds = jnp.std(collected_sample[:,1:],axis=0)

    keys = z.attrs['features'][1:]

    stats = {
        k: {'mean': m, 'std': s}
         for k, m, s in zip(keys, means.tolist(), stds.tolist())
     }

    return stats

stats  = compute_stats('../data/01_processed/ssl_sets/ob007.zarr')

print(stats)
# def compute_log_normalization_stats(df, features, epsilon=1):
#     means = {col: (df[col].explode() + epsilon).log().mean() for col in features}
#     stds = {col: (df[col].explode() + epsilon).log().explode().std() for col in features}
#     output_dict = {'log_norm':{
#         'means': means,
#         'stds': stds
#     }}
#     return output_dict


# def main():
#     parser = argparse.ArgumentParser(
#         description="Generates normalization statistics based on the train partition of methylation data"
#     )
#     parser.add_argument('-i', '--input_path',
#                         type = str,
#                         required=True,
#                         help="Input filepath to the training data")
#     parser.add_argument('-o', '--output_path',
#                         type=str,
#                         required=True,
#                         help="Path to output file (including filename)")
#     parser.add_argument('-c', '--config_path',
#                         type=str,
#                         help='path to config file')
#     parser.add_argument('-t', '--truncate',
#                         action='store_true',
#                         help='If specified, only uses first 10_000 samples')
    
#     args = parser.parse_args()
#     print('began running compute stats')
#     config = parse_yaml(args.config_path)
#     q = (
#         pl.scan_parquet(os.path.expanduser(args.input_path),
#                         schema = config['data']['schema']
#                         )
#                     )

#     df = q.collect()
#     print('collected df')
#     exp_outpath = args.output_path

#     kinetics_features = config['data']['kinetics_features']

#     stats_dict = compute_log_normalization_stats(df, kinetics_features)
#     print('computed stats')
#     os.makedirs(os.path.dirname(exp_outpath), exist_ok=True)

#     with open(exp_outpath, 'w') as f:
#         yaml.dump(stats_dict, f, indent=4)

#     print(f"Normalization stats saved to {exp_outpath}")
#     print(yaml.dump(stats_dict, indent=4))


# # if __name__ == "__main__":
#     main()
