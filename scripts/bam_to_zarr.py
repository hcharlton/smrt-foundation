import pysam
import argparse
import sys
import zarr
import numpy as np
from smrt_foundation.utils import parse_yaml

from numcodecs import blosc
blosc.set_nthreads(6)

# --- Profile boilerplate begin ---
import os
import builtins
import atexit

if os.environ.get('TimeLINE_PROFILE'):
    # only imports line_profiler if the env var is set
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    # Define the decorator to wrap functions
    def profile(func):
        return lp(func)
        
    # save log
    def save_profile():
        # filename
        lp.dump_stats('profile_output.lprof') 
        print("\n[Profiler] Stats saved to 'profile_output.lprof'")
    
    atexit.register(save_profile)
else:
    def profile(func):
        return func
builtins.profile = profile
# --- Profile boilerplate end ---

@profile
def _process_read(read, tags):
    """
    Processes a single pysam.AlignmentRead.
    Checks for required tags and extracts all full-length tag data.
    """
    if not all(read.has_tag(tag) for tag in set(tags)-{'seq', 'qual'}):
        return None
    read_data = {}
    for tag in set(tags) - {'seq', 'qual'}:
        tag_data = read.get_tag(tag) if read.has_tag(tag) else None
        read_data[tag] = tag_data
    read_data |= {
        'seq': read.query_sequence,
        'qual': np.frombuffer(read.qual.encode('ascii'), dtype=np.uint8) - 33
    }
    
    # return None if any tags are missing
    if any(read_data[tag] is None or len(read_data[tag]) == 0 for tag in tags):
        return None
    # return None if the sequence is not consistent length across tags
    if len(set([len(v) for k, v in read_data.items()])) != 1:
        return None
    # return a dictionary with data from the read
    return {
        "name": read.query_name,
        "data": read_data,
        "seq_len": read.query_length
    }

def _check_tags(bam_path, tags, n_reads=100, threshold=0.8):
    """
    Checks the first n_check reads. 
    Raises ValueError if ANY requested tag is missing in > threshold % of reads.
    """
    if not tags:
        return

    tag_counts = {t: 0 for t in tags}
    reads_checked = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads:
                break
            reads_checked += 1
            for t in tags:
                if read.has_tag(t):
                    tag_counts[t] += 1
    
    if reads_checked == 0:
        raise ValueError("BAM file appears empty.")

    # value error if any of the tags don't meet the threshold
    for t, count in tag_counts.items():
        missing_rate = 1.0 - (count / reads_checked)
        if missing_rate > threshold:
             raise ValueError(
                 f"Tag '{t}' is missing in {missing_rate:.1%} of the first {reads_checked} reads. "
                 f"Threshold is {threshold:.0%}. Aborting."
             )
    
    print(f"Validation successful: All requested tags present in >{1-threshold:.0%} of checked reads.")

@profile
def bam_to_zarr(bam_path: str, zarr_path: str, n_reads: int, optional_tags: list, config: dict):
    seq_map = config['data']['token_map']
    fixed_tags = ['seq','qual'] + sorted(list(config['data']['kinetics_features']))
    variable_tags = sorted([t for t in optional_tags if t not in fixed_tags])
    out_columns = fixed_tags+variable_tags
    n = len(out_columns) 
    tag_to_idx = {tag: i for i, tag in enumerate(out_columns)}
    _check_tags(bam_path=bam_path, tags=set(out_columns)-{'seq','qual'}, n_reads=20) # check that the requested tags exist -> exit if they don't

    def _generate_array(tag, read_data):
        """
        Helper function to convert a python list to a np array
        """
        if tag == 'seq':
            return lookup_table[np.frombuffer(read_data[tag].upper().encode('ascii'), dtype=np.uint8)]
        else:
            return np.array(read_data[tag], dtype=np.uint8)
        
    # transfer to numpy byte array
    lookup_table = np.zeros(128, dtype=np.uint8)
    for base, val in seq_map.items():
        if len(base) == 1:
            lookup_table[ord(base)] = val
    
    counters = { "reads_processed": 0, "reads_skipped": 0, }

    #initialize zarr object
    root = zarr.create_group(store=zarr_path)
    root.attrs['features'] = out_columns
    root.attrs['tag_to_idx'] = tag_to_idx
    shard_size_bases = 400_000_000 # unit: bases
    chunk_size_bases = 40_000_000 #20_000_000 -> fast for write
    # initialize the arrays
    z_data = root.create_array(name = 'data', shape=(0, n), chunks=(chunk_size_bases, n), shards=(shard_size_bases, n), dtype='uint8', overwrite=True)
    z_indptr = root.create_array(name = 'indptr', shape=(1,), chunks=(shard_size_bases,), dtype='uint64', overwrite=True)
    z_indptr[0] = 0 # initialize the start of the index pointers
    total_len=0
    # Batching info. This is important for zarr write performance. Writes should be larger than a shard
    batch_size_reads = shard_size_bases/2_000 # Largely empirical. Note that the unit of this is reads... which conservatively are 10-20k bases 
    batch_data = []
    batch_indptr = []
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=5) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads != 0:
                break
            
            read_dict = _process_read(read, tags=out_columns)
            
            if read_dict is None:
                counters["reads_skipped"] += 1
                continue
            counters["reads_processed"] += 1 
            read_array = np.stack(([_generate_array(tag, read_dict['data']) for tag, idx in tag_to_idx.items()]), axis=1)
            read_len = read_array.shape[0]
            total_len += read_len
            batch_data.append(read_array)
            batch_indptr.append(total_len)

            if len(batch_data) >= batch_size_reads:
                stacked_batch = np.concatenate(batch_data, axis=0, dtype='uint8')
                z_data.append(stacked_batch, axis=0)
                z_indptr.append(np.array(batch_indptr, dtype='uint64'))
                batch_data = []
                batch_indptr = []
        if batch_data:
            stacked_batch = np.concatenate(batch_data, axis=0, dtype='uint8')
            z_data.append(stacked_batch, axis=0)
            z_indptr.append(np.array(batch_indptr, dtype='uint64'))
    

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
