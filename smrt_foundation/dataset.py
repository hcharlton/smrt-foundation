import torch
import polars as pl
import glob
import os
import numpy as np
import pyarrow.parquet as pq
from collections import OrderedDict
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset


class ShardedMemmapDataset(Dataset):
    def __init__(self, data_dir, cache_size=100):
        expanded_dir = os.path.expandvars(data_dir)
        self.shard_paths = sorted(glob.glob(os.path.join(expanded_dir, "*.npy")))
        first_shard = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = first_shard.shape[0]
        last_shard = np.load(self.shard_paths[-1], mmap_mode='r')
        self.total_len = ((len(self.shard_paths) - 1) * self.shard_size) + last_shard.shape[0]
        self.cache_size = cache_size
        self.memmaps = OrderedDict()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        if shard_idx not in self.memmaps:
            if len(self.memmaps) >= self.cache_size:
                self.memmaps.popitem(last=False)
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
        else:
            self.memmaps.move_to_end(shard_idx)
        return torch.from_numpy(np.array(self.memmaps[shard_idx][local_idx])).bfloat16()
    

def compute_log_normalization_stats(df, features, epsilon=1):
    means = {col: (df[col].explode() + epsilon).log().mean() for col in features}
    stds = {col: (df[col].explode() + epsilon).log().explode().std() for col in features}
    return means, stds

class LegacyMethylDataset(IterableDataset):
    def __init__(self, data_path, means, stds, context, restrict_row_groups=0, single_strand=False, inference=False):
        super().__init__()
        self.data_path = Path(data_path)
        self.means, self.stds = means, stds
        self.context = context
        self.single_strand = single_strand
        self.inference = inference
        self.restrict = restrict_row_groups

        self.kin_feats = ['fi', 'fp', 'ri', 'rp']
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.comp_map = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

        try:
            meta = pq.read_metadata(self.data_path)
            self.n_groups = meta.num_row_groups
            use_groups = min(self.restrict, self.n_groups) if self.restrict else self.n_groups

            # fast row count
            n_rows = sum(meta.row_group(i).num_rows for i in range(use_groups))
            self.len = n_rows * (2 if single_strand else 1)
        except Exception:
            print(f'Failed to read parquet: {self.data_path}')
            self.n_groups, self.len = 0, 0

    def __len__(self):
        return self.len

    def _process_batch(self, df):
      # seq
        seq_arr = np.stack(
            df['seq'].str.split("")
            .list.eval(pl.element().replace_strict(self.vocab, default=4))
            .to_numpy()
        )
        seq_t = torch.tensor(seq_arr, dtype=torch.long)

        # kinetics
        kin_list = []
        for k in self.kin_feats:
            vals = df[k].to_numpy() # (N, L)
            vals = (np.log(vals + 1) - self.means[k]) / self.stds[k]
            kin_list.append(vals)
        kin_t = torch.tensor(np.stack(kin_list, axis=1), dtype=torch.bfloat16)

        # mask, labels, etc (note that there is no masked data in the downstream set, so it's all zeros here)
        mask = torch.zeros((seq_t.shape[0], seq_t.shape[1], 1), dtype=torch.bfloat16)
        labels = torch.tensor(df['label'].to_numpy(), dtype=torch.long) if not self.inference else None
        r_names, pos = df['read_name'].to_list(), df['cg_pos'].to_list()

        # construct forward sample
        # Seq (N, L, 1) + Kin (N, 2, L)->(N, L, 2) + Mask (N, L, 1) = (N, L, 4)
        fwd_data = torch.cat([
            seq_t.unsqueeze(-1).to(torch.bfloat16),
            kin_t[:, 0:2].permute(0, 2, 1),
            mask
        ], dim=2)

        # construct reverse data
        rev_data = None
        if self.single_strand:
            # rev_seq_t = torch.flip(self.comp_map.to(seq_t.device)[seq_t], dims=[1])
            rev_seq_t = torch.flip(self.comp_map[seq_t], dims=[1])
            # Kin: slice 2:4, flip time (dim 2), permute channels
            rev_kin = torch.flip(kin_t[:, 2:4], dims=[2]).permute(0, 2, 1)
            rev_data = torch.cat([
                rev_seq_t.unsqueeze(-1).to(torch.bfloat16),
                rev_kin,
                mask
            ], dim=2)

        # yield
        for i in range(len(df)):
            # forward
            strand_name = 'fwd' if self.single_strand else 'ds'
            item_fwd = {
                'data': fwd_data[i],
                'metadata': {'read_name': r_names[i], 'position': pos[i], 'strand': strand_name}
            }
            if labels is not None: item_fwd['label'] = labels[i]
            yield item_fwd

            # reverse
            if rev_data is not None:
                item_rev = {
                    'data': rev_data[i],
                    'metadata': {'read_name': r_names[i], 'position': pos[i], 'strand': 'rev'}
                }
                if labels is not None: item_rev['label'] = labels[i]
                yield item_rev
            else:
              continue

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        valid_groups = min(self.restrict, self.n_groups) if self.restrict else self.n_groups
        indices = np.arange(valid_groups)

        if worker:
            indices = np.array_split(indices, worker.num_workers)[worker.id]

        pqf = pq.ParquetFile(self.data_path)
        for i in indices:
            # array cast
            df = pl.from_arrow(pqf.read_row_group(i)).with_columns([
                pl.col(c).list.to_array(self.context) for c in self.kin_feats
            ])
            yield from self._process_batch(df)