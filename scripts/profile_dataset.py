import time
from torch.utils.data import DataLoader
import yaml
import os
import sys
from tqdm import tqdm

module_path = os.path.abspath("/dcai/users/chache/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LabeledMemmapDataset, ChunkedRandomSampler
from smrt_foundation.normalization import ZNorm

def profile_cache(dataset, sampler=None, shuffle=False, num_batches=1_000, num_workers=0):
    dataset.cache_hits = 0
    dataset.cache_misses = 0
    loader = DataLoader(dataset, batch_size=256, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    
    t0 = time.time()
    for i, _ in enumerate(tqdm(loader)):
        if i >= num_batches:
            break
    t1 = time.time()
    
    total = dataset.cache_hits + dataset.cache_misses
    hit_rate = dataset.cache_hits / total if total > 0 else 0
    
    print(f"Shuffle: {shuffle:<5}, Sampler: {True if sampler else False} | Time: {t1-t0:.2f}s | Hit Rate: {hit_rate:.2%} | Hits: {dataset.cache_hits} | Misses: {dataset.cache_misses}")

with open('./configs/supervised.yaml', 'r') as f:
        config = yaml.safe_load(f)

limit=256*10_000
tmp_ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'))
norm_fn = ZNorm(tmp_ds)
ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), norm_fn = norm_fn)

sampler = ChunkedRandomSampler(ds, 2048, shuffle_within=True)

profile_cache(ds, shuffle=False)
profile_cache(ds, shuffle=True)
profile_cache(ds, sampler=sampler)