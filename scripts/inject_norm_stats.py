# import polars as pl
# import os
# import yaml
import jax.numpy as jnp
import tensorstore as ts
import concurrent.futures
import zarr
import argparse
# from smrt_foundation.utils import parse_yaml


def read_chunk(args):
    zarr_path, chunk_idx,  data_step, chunk_len = args
    z = zarr.open(zarr_path, mode='r')
    start = chunk_idx*chunk_len
    end = min(start+chunk_len, z['data'].shape[0])
    return z['data'][start:end:data_step]

def compute_stats(zarr_path, chunk_stride, idx_stride, num_threads):
    z = zarr.open(zarr_path, mode='r')
    data = z['data']

    total_chunks = data.cdata_shape[0]
    chunk_len = data.chunks[0]

    indices = range(0, total_chunks, chunk_stride)
    tasks = [(zarr_path, i, idx_stride, chunk_len) for i in indices]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        sampled_chunks = list(executor.map(read_chunk, tasks))

    collected_log_sample = jnp.log1p(jnp.concatenate(sampled_chunks))

    means = jnp.mean(collected_log_sample[:,1:],axis=0)
    stds = jnp.std(collected_log_sample[:,1:],axis=0)

    keys = z.attrs['features'][1:]

    stats = {
        k: {'mean': m, 'std': s}
         for k, m, s in zip(keys, means.tolist(), stds.tolist())
     }
    return stats

def inject_stats(zarr_path:str, stats:dict):
    z = zarr.open(zarr_path, mode='r+')
    z.attrs['log_norm'] = stats
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generates normalization statistics based on the train partition of methylation data"
    )
    parser.add_argument('--input_path',
                        type = str,
                        required=True,
                        help="Input filepath to the training data")
    parser.add_argument('--chunk_stride',
                        type=int,
                        required=True,
                        help='Stride by which to process chunks (greater number skips more chunks)')
    parser.add_argument('--idx_stride',
                        type=int,
                        required=True,
                        help='Stride by which to iterate through a chunk (greater number skips more indices)')
    parser.add_argument('--num_threads',
                        type=int,
                        required=True,
                        help='Number of threads for multiprocessing')

    args = parser.parse_args()
    stats = compute_stats(zarr_path = args.input_path,
                          chunk_stride = args.chunk_stride,
                          idx_stride = args.idx_stride,
                          num_threads = args.num_threads)
    inject_stats(args.input_path, stats)


if __name__ == "__main__":
    main()
