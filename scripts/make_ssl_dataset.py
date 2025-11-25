import pysam
import polars as pl
import argparse
import sys
import os
import yaml
from smrt_foundation.dataset import SCHEMA
from smrt_foundation.utils import parse_yaml




### Purpose ###
# converts a BAM file into a parquet file of fixed-size windows 
# for transformer or general context-based ssl training.

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
    seq_len = len(read.query_sequence)
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

def bam_to_df(bam_path: str, denomination: str, n_reads: int, context: int, optional_tags: list, config: dict):
    per_base_tags = config['data']['per_base_tags']
    required_tags = config['data']['kinetics_features']
    tags_union = required_tags.union(set(optional_tags))

    seq_map = config['data']['token_map']

    final_cols = ["read_name", "read_pos", "seq", "qual"] + list(tags_union)
    col_data = {key: [] for key in final_cols}
    
    counters = { "reads_processed": 0, "reads_skipped": 0, "windows_created": 0 }

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads != 0:
                break
            
            processed_read = _process_read(read, optional_tags, required_tags)
            
            if processed_read is None:
                counters["reads_skipped"] += 1
                continue
            
            seq = processed_read["seq"].upper()
            qual_values = processed_read["qual"]
            read_tag_data = processed_read["tag_data"]
            L = processed_read["seq_len"]
            int_seq = [seq_map.get(base, 0) for base in seq]

            for win_start in range(0, L - context + 1, context):
                win_end = win_start + context
                
                # Calculate reverse indices 
                rev_win_start = L - win_end
                rev_win_end = L - win_start

                col_data["read_name"].append(processed_read["name"])
                col_data["read_pos"].append(win_start)

                col_data["seq"].append(int_seq[win_start:win_end])

                col_data["qual"].append(qual_values[win_start:win_end])

                for tag, values in read_tag_data.items():
                    if values is None:
                        col_data[tag].append(None) 
                        continue
                    
                    if tag in per_base_tags:
                        sliced_values = values[rev_win_start:rev_win_end] if tag in {"ri", "rp"} else values[win_start:win_end]
                        col_data[tag].append(sliced_values)
                    else: # Non-per-base tags like 'np'
                        col_data[tag].append(values)

                counters["windows_created"] += 1
            
            counters["reads_processed"] += 1

    print("--- Debugging Counters ---")
    for key, value in counters.items():
        print(f"{key:<25}: {value}")
    print("--------------------------")

    if counters["windows_created"] == 0:
        print("no valid windows were found. Returning empty dataframe.")
        return pl.DataFrame()
    
    final_schema = {k: v for k, v in SCHEMA.items() if k in col_data and col_data[k]}
    df = pl.DataFrame(col_data, schema=final_schema).with_columns(
        pl.lit(denomination).alias('sample')
    )
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Processes the first N reads or 0 for all of the methylation dataset." \
        "Takes context size as a parameter. Outputs a parquet file containing 1 sample per row."
    )
    parser.add_argument('--input_path',
                    type=str,
                    required=True,
                    help="Path to the input BAM file.")
    
    parser.add_argument('--n_reads',
                        type=int,
                        required=True,
                        help="Number of reads to process. 0 for all reads.")
    parser.add_argument('--context',
                        type=int,
                        required=True,
                        help='The fixed window/context size for the transformer (e.g., 2048).')
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


    df = bam_to_df(bam_path=os.path.expanduser(args.input_path),
                   denomination=args.denomination,
                   n_reads=args.n_reads, 
                   context=args.context,
                   optional_tags=args.optional_tags,
                   config=config)

    df.write_parquet(args.output_path)

if __name__ == "__main__":
    main()