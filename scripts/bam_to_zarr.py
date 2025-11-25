import pysam
import argparse
import sys
import zarr
import numpy as np
from smrt_foundation.utils import parse_yaml

from numcodecs import blosc
blosc.set_nthreads(4)

import os
import builtins
import atexit

# --- Conditional Profiling Setup ---
if os.environ.get('TimeLINE_PROFILE'):
    # Only imports line_profiler if the env var is set
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    # Define the decorator to wrap functions
    def profile(func):
        return lp(func)
        
    # Save data when the script exits (successfully or via error)
    def save_profile():
        # You can change the filename here if needed
        lp.dump_stats('profile_output.lprof') 
        print("\n[Profiler] Stats saved to 'profile_output.lprof'")
    
    atexit.register(save_profile)
else:
    # No-op decorator: runs code at native speed
    def profile(func):
        return func

# Inject into builtins so @profile is available everywhere
builtins.profile = profile
# -----------------------------------

@profile
def _process_read(read, optional_tags, required_tags, per_base_tags):
    """
    Processes a single pysam.AlignmentRead.
    Checks for required tags and extracts all full-length tag data.
    """
    if not all(read.has_tag(tag) for tag in required_tags):
        return None
    tags_union = required_tags.union(set(optional_tags))
    read_data = {}
    for tag in tags_union:
        tag_data = read.get_tag(tag) if read.has_tag(tag) else None
        read_data[tag] = tag_data

    # return None if any required tags are missing
    if any(read_data[tag] is None or len(read_data[tag]) == 0 for tag in required_tags):
        return None

    # return None if qualities is not the same length as seq
    seq_len = read.query_length
    if len(read.query_qualities) != seq_len:
        return None

    # return None if any of the tag data lengths that do not match seq
    for tag in tags_union:
        if tag in per_base_tags and read_data[tag] is not None:
            if len(read_data[tag]) != seq_len:
                return None 
    # retun a dictionary with data from the read
    return {
        "name": read.query_name,
        "seq": read.query_sequence,
        "qual": list(read.query_qualities),
        "tag_data": read_data,
        "seq_len": seq_len
    }
@profile
def bam_to_zarr(bam_path: str, zarr_path: str, n_reads: int, optional_tags: list, config: dict):
    per_base_tags = set(config['data']['per_base_tags'])
    required_tags = set(config['data']['kinetics_features'])
    tags_union = required_tags.union(set(optional_tags))

    seq_map = config['data']['token_map']
    
    counters = { "reads_processed": 0, "reads_skipped": 0, }

    n = len(per_base_tags.intersection(tags_union)) + 2 # add one for seq and one for qual, which is not a tag

    root = zarr.create_group(store=zarr_path)
    shard_size = 10_000_000
    chunk_size = 100_000
    z_data = root.create_array(name = 'data', shape=(n, 0), chunks=(n, chunk_size), shards=(n, shard_size), dtype='uint8', overwrite=True)
    z_indptr = root.create_array(name = 'indptr', shape=(1,), chunks=(shard_size,), shards=(shard_size,), dtype='uint32', overwrite=True)
    z_indptr[0] = 0 # initialize the start of the index pointers
    total_len=0
    # Batching info. This is important for zarr write performance. Writes should be larger than a shard
    batch_size = (shard_size/17_000)*2 # approx reads per 10 shards -> We write 10 shards per batch
    batch_data = []
    batch_indptr = []
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads != 0:
                break
            
            read_data = _process_read(read, optional_tags, required_tags, per_base_tags=per_base_tags)
            
            if read_data is None:
                counters["reads_skipped"] += 1
                continue
            counters["reads_processed"] += 1 
            tag_dict = read_data["tag_data"]
            # seq_len = read_data["seq_len"]
            seq_str = read_data["seq"].upper()

            seq_vec = np.array([seq_map.get(base, 0) for base in seq_str], dtype=np.uint8)
            qual_vec = np.array(read_data["qual"], dtype = np.uint8)
            fi_vec = np.array(tag_dict['fi'], dtype=np.uint8)
            fp_vec = np.array(tag_dict['fp'], dtype=np.uint8)
            ri_vec = np.array(tag_dict['ri'], dtype=np.uint8)
            rp_vec = np.array(tag_dict['rp'], dtype=np.uint8)

            read_array = np.stack([seq_vec, 
                                   qual_vec, 
                                   fi_vec, 
                                   fp_vec,
                                   ri_vec, 
                                   rp_vec], axis=0)
            read_len = read_array.shape[1]
            total_len += read_len
            print(read_len)
            batch_data.append(read_array)
            batch_indptr.append(total_len)

            if len(batch_data) >= batch_size:
                stacked_batch = np.concatenate(batch_data, axis=1, dtype='uint8')
                z_data.append(stacked_batch, axis=1)
                z_indptr.append(np.array(batch_indptr, dtype='uint32'))
                batch_data = []
                batch_indptr = []
        if batch_data:
            stacked_batch = np.concatenate(batch_data, axis=1, dtype='uint8')
            z_data.append(stacked_batch, axis=1)
            z_indptr.append(np.array(batch_indptr, dtype='uint32'))
    

    print("--- Debugging Counters ---")
    for key, value in counters.items():
        print(f"{key:<25}: {value}")
    print("--------------------------")

    if counters["reads_processed"] == 0:
        print("no valid reads were found.")

    
@profile
def main():
    parser = argparse.ArgumentParser(
        description="Processes the first N reads or 0 for all of the  dataset." \
        "Outputs a zarr file"
    )
    parser.add_argument('--input_path',
                    type=str,
                    required=True,
                    help="Path to the input BAM file.")
    
    parser.add_argument('--n_reads',
                        type=int,
                        required=True,
                        help="Number of reads to process. 0 for all reads.")
    parser.add_argument('--optional_tags',
                        type=str,
                        nargs='*', 
                        default=[],
                        help='Space-separated list of optional tags to extract (e.g., np sm sx).')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help="path for output file")
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Path the config file for model configuration')
    parser.add_argument('--denomination',
                        type=str,
                        required=True,
                        help='name of sample to be used in dataframe')
    args = parser.parse_args()
    if args.n_reads < 0:
        print("Error: n_reads should be positive or 0 (to indicate all reads).")
        sys.exit(1)
    
    config = parse_yaml(args.config)
    
    bam_to_zarr(bam_path=os.path.expanduser(args.input_path),
                zarr_path = args.output_path,
                n_reads=args.n_reads, 
                optional_tags=args.optional_tags,
                config=config)


if __name__ == "__main__":
    main()