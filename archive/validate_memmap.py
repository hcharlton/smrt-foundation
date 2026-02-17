import os
import sys
import json
import argparse
import logging
import numpy as np
from smrt_foundation.utils import parse_yaml

def verify_output(data_dir, config_path):
    # Configure logging to file
    config = parse_yaml(config_path)
    log_path = os.path.join(data_dir, "validation.log")
    logging.basicConfig(
        filename=log_path,
        filemode='w',  # Overwrite mode. Use 'a' to append.
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
  
    logging.info(f"Verifying data in: {data_dir}")

    # --- load metadata ---
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        logging.error("dataset_schema.json not found")
        sys.exit(1)

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    logging.info("Schema loaded successfully.")
    features = schema['features']
    seq_idx = features.index('seq')
    mask_idx = features.index('mask')
    logging.info(f"Mapped 'seq' to col {seq_idx} and 'mask' to col {mask_idx}")

    # --- load first shard ---
    shard_path = os.path.join(data_dir, "shard_00002.npy")
    if not os.path.exists(shard_path):
        logging.error("shard_00002.npy not found!")
        sys.exit(1)

    data = np.load(shard_path)
    logging.info(f"Loaded shard_00002.npy. Shape: {data.shape} | Dtype: {data.dtype}")

    if data.dtype != np.float16:
        logging.warning(f"Data is {data.dtype}, expected float16.")

    # --- TEST 1: Column Integrity (Seq vs Signal) ---
    logging.info("[TEST 1] Check Column Type Integrity")

    seq_col = data[:, :, seq_idx].flatten()
    valid_seq = seq_col[data[:, :, mask_idx].flatten() == 0.0]

    is_discrete = np.all(np.abs(valid_seq - np.round(valid_seq)) < 0.1)
    if is_discrete:
        unique_vals = np.unique(np.round(valid_seq))
        logging.info(f"'seq' column contains discrete values: {unique_vals}")
    else:
        logging.error("'seq' column contains non-integers — indicates it was normalized")
        logging.info(f"Sample values: {valid_seq[:10]}")

    # --- TEST 2: Normalization Stats ---
    logging.info("[TEST 2] Check Normalization (Target: Mean~0, Std~1)")

    for i, feat in enumerate(features):
        if feat in ['seq', 'mask']: continue

        col_data = data[:, :, i]
        valid_data = col_data[data[:, :, mask_idx] == 0.0]

        curr_mean = np.mean(valid_data, dtype = np.float64)
        curr_std = np.std(valid_data, dtype = np.float64)

        if abs(curr_mean) < 0.2 and abs(curr_std - 1.0) < 0.2:
            logging.info(f"Feature '{feat}': Mean={curr_mean:.3f}, Std={curr_std:.3f}")
        else:
            logging.warning(f"Feature '{feat}': Mean={curr_mean:.3f}, Std={curr_std:.3f}")

    # --- TEST 3: Reverse Complement Logic ---
    logging.info("[TEST 3] Check Reverse Complement Logic")
    nuc_to_int = config['data']['token_map']
    rc_map = config['data']['rc_map']
    rc_lookup = np.zeros(len(nuc_to_int), dtype=int)
    for base, idx in nuc_to_int.items():
        comp_base = rc_map[base]
        rc_lookup[idx] = nuc_to_int[comp_base]

    fwd_row = data[0]
    rev_row = data[1]

    fwd_len = np.sum(fwd_row[:, mask_idx]==0.0)
    rev_len = np.sum(rev_row[:, mask_idx]==0.0)

    if fwd_len != rev_len:
         logging.error(f"Length mismatch. Fwd: {fwd_len}, Rev: {rev_len}")
    else:
         logging.info(f"success: sequence lengths match ({int(fwd_len)} bp).")

    seq_fwd_valid = fwd_row[:int(fwd_len), seq_idx]
    seq_rev_valid = rev_row[:int(fwd_len), seq_idx]

    manual_flip = np.flip(seq_fwd_valid)
    manual_rc = rc_lookup[manual_flip.astype(int)]
    matches = (seq_rev_valid == manual_rc)
    # manual_rc = manual_flip.copy()
    # mask_bases = manual_flip < 3.5
    # manual_rc[mask_bases] = 3.0 - manual_flip[mask_bases]

    # matches = np.abs(seq_rev_valid - manual_rc) < 0.1
    match_pct = np.mean(matches) * 100

    if match_pct > 99.9:
        logging.info("success on reverse complement logic.")
        logging.info(f"Sample Fwd: {seq_fwd_valid[:5]} ...")
        logging.info(f"Sample Rev: {seq_rev_valid[:5]} ...")
    else:
        logging.error(f"Reverse complement match pct {match_pct:.2f}% match.")
        logging.info(f"Debug Fwd (End, flipped): {manual_flip[:5]}")
        logging.info(f"Debug Expected RC:        {manual_rc[:5]}")
        logging.info(f"Debug Actual Rev File:    {seq_rev_valid[:5]}")

    # --- TEST 4: Padding Check ---
    logging.info("[TEST 4] Check Padding")
    padding_mask = data[:,:,mask_idx] == 1.0
    if np.sum(padding_mask) > 0:
        padding_features = data[padding_mask, :-1]
        non_zero_padded_features = np.sum(padding_features != 0.0)
        if non_zero_padded_features == 0:
            logging.info("Success: All padded features contain pure zeros.")
        else:
            logging.error(f"Failure: found {non_zero_padded_features} non-zero values in feature section of padding")
            bad_indices = np.where(padding_features != 0.0)
            logging.info(f"sample non-zero pad value: {padding_features[bad_indices][0]}")
    else:
        logging.info("No padding found in this shard")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    verify_output(args.input_path, args.config_path)
