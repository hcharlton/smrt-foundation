import os
import json
import numpy as np
import argparse
import sys

def verify_output(data_dir):
    print(print("Verifying data in: {data_dir}\n")
    
    # --- 1. Load Metadata ---
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        print ("error: dataset_schema.json not found!")
        sys.exit(1)
        
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    print("Schema loaded successfully.")
    features = schema['features']
    seq_idx = features.index('seq')
    mask_idx = features.index('mask')
    print(f"   Mapped 'seq' to col {seq_idx} and 'mask' to col {mask_idx}")

    # --- 2. Load First Shard ---
    shard_path = os.path.join(data_dir, "shard_00000.npy")
    if not os.path.exists(shard_path):
        print("error: shard_00000.npy not found!")
        sys.exit(1)
        
    data = np.load(shard_path)
    print(f"loaded shard_00000.npy. Shape: {data.shape} | Dtype: {data.dtype}")
    
    if data.dtype != np.float16:
        print(f"warning: Data is {data.dtype}, expected float16.")

    # --- TEST 1: Column Integrity (Seq vs Signal) ---
    print("\n[TEST 1] Column Type Integrity")
    
    # Check Sequence Column: Should be essentially integers (0.0, 1.0, 2.0, 3.0, 4.0)
    seq_col = data[:, :, seq_idx].flatten()
    valid_seq = seq_col[data[:, :, mask_idx].flatten() == 1.0] # Only check valid data
    
    # Are values close to integers?
    # We allow small float16 epsilon, but they should be distinct levels.
    is_discrete = np.all(np.abs(valid_seq - np.round(valid_seq)) < 0.1)
    if is_discrete:
        unique_vals = np.unique(np.round(valid_seq))
        print(f"'seq' column contains discrete values: {unique_vals}")
    else:
        print(f" 'seq' column contains non-integers — indicates it was normalized")
        print(f"      Sample values: {valid_seq[:10]}")

    # --- TEST 2: Normalization Stats ---
    print("\n[TEST 2] Statistical Normalization Check (Target: Mean~0, Std~1)")
    
    for i, feat in enumerate(features):
        if feat in ['seq', 'mask']: continue
        
        col_data = data[:, :, i]
        valid_data = col_data[data[:, :, mask_idx] == 1.0] # Ignore padding
        
        curr_mean = np.mean(valid_data)
        curr_std = np.std(valid_data)
        
        status = "success" if abs(curr_mean) < 0.2 and abs(curr_std - 1.0) < 0.2 else "warning "
        print(f"   {status} Feature '{feat}': Mean={curr_mean:.3f}, Std={curr_std:.3f}")

    # --- TEST 2: Reverse Complement Logic ---
    print("\n[TEST 3] Reverse Complement Alignment")
    # We assume Row 0 is FWD and Row 1 is REV
    
    fwd_row = data[0]
    rev_row = data[1]
    
    # 1. Check Mask Alignment
    # The valid length (number of 1s in mask) should be identical
    fwd_len = np.sum(fwd_row[:, mask_idx])
    rev_len = np.sum(rev_row[:, mask_idx])
    
    if fwd_len != rev_len:
         print(f"error: length mismatch. Fwd: {fwd_len}, Rev: {rev_len}")
    else:
         print(f"success: sequence lengths match ({int(fwd_len)} bp).")

    # 2. Verify Sequence Transformation (Time Flip + Complement)
    # Extract only valid sequence parts
    seq_fwd_valid = fwd_row[:int(fwd_len), seq_idx]
    seq_rev_valid = rev_row[:int(fwd_len), seq_idx]
    
    # Manually calculate RC from FWD
    # Step 1: Flip Time
    manual_flip = np.flip(seq_fwd_valid)
    
    # Step 2: Complement (3.0 - x) for bases < 4
    manual_rc = manual_flip.copy()
    mask_bases = manual_flip < 3.5 # Ignore 'N' (4.0)
    manual_rc[mask_bases] = 3.0 - manual_flip[mask_bases]
    
    # Compare with what is in the file
    matches = np.abs(seq_rev_valid - manual_rc) < 0.1
    match_pct = np.mean(matches) * 100
    
    if match_pct > 99.9:
        print(f"   success on reverse complement logic.")
        print(f"           Sample Fwd: {seq_fwd_valid[:5]} ...")
        print(f"           Sample Rev: {seq_rev_valid[:5]} ... (should be RC of end of Fwd)")
    else:
        print(f"   error on reverse complement: match pct {match_pct:.2f}% match.")
        print("          Debug:")
        print(f"         Fwd (End, flipped): {manual_flip[:5]}")
        print(f"         Expected RC:        {manual_rc[:5]}")
        print(f"         Actual Rev File:    {seq_rev_valid[:5]}")

    # --- TEST 4: Padding Check ---
    print("\n[TEST 4] Padding Safety")
    # Check regions where mask is 0. All other features MUST be 0.
    
    invalid_mask = data[:, :, mask_idx] == 0.0
    if np.sum(invalid_mask) > 0:
        leakage = np.sum(np.abs(data[invalid_mask]))
        if leakage == 0.0:
            print("success: padding is clean (all 0s outside data).")
        else:
            print(f"error: non-valid padding found in padding zones. sum of absolute values: {leakage}")
    else:
        print("padding not present indicating a full seqeuence.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    verify_output(args.data_dir)
