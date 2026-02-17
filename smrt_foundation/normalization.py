# smrt_foundation/normalization.py
import numpy as np

def build_rc_lookup(config):
    """
    Creates a numpy lookup table for RC conversion based on config maps.
    Returns: np.array where index=input_token, value=rc_token (requires ints)
    """
    token_map = config['data']['token_map']
    rc_map = config['data']['rc_map']
    
    max_token = max(token_map.values())
    
    lookup = np.arange(max_token + 1, dtype=np.int8)
    
    for base, idx in token_map.items():
        if base in rc_map:
            comp_base = rc_map[base]
            if comp_base in token_map:
                lookup[idx] = token_map[comp_base]
                
    return lookup
    
### MAD normalization
def normalize_read_mad(read_data, is_continuous_mask, eps=1e-6):
    """
    MAD normalization of a single read on the continous features
    
    read_data: array of data for one read
    is_continuous_mask: boolean mask to index only the continuous features 
                        (masks out the categorical features)
    eps: Description
    """
    np.log1p(read_data, out=read_data, where=is_continuous_mask)
    x = read_data[:, is_continuous_mask]
    x_median = np.median(x, axis=0)
    mad = np.median(np.abs(x - x_median), axis=0)
    mad = np.where(mad < eps, 1.0, mad)
    x_norm = (x - x_median) / (mad * 1.4826)
    read_data[:, is_continuous_mask] = x_norm

    return read_data

