import torch
import polars as pl
import glob
import os
import numpy as np
import pyarrow.parquet as pq
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, Sampler

class ChunkedRandomSampler(Sampler):
    def __init__(self, data_source, chunk_size, shuffle_within=True):
        self.num_samples = len(data_source)
        self.chunk_size = chunk_size
        self.shuffle_within = shuffle_within

    def __iter__(self):
        chunks = [range(i, min(i + self.chunk_size, self.num_samples)) 
                  for i in range(0, self.num_samples, self.chunk_size)]
        
        for chunk_idx in torch.randperm(len(chunks)).tolist():
            chunk = chunks[chunk_idx]
            if self.shuffle_within:
                for idx in torch.randperm(len(chunk)).tolist():
                    yield chunk[idx]
            else:
                yield from chunk

    def __len__(self):
        return self.num_samples


class ShardedMemmapDataset(Dataset):
    def __init__(self, data_dir, cache_size=100, limit=0):
        expanded_dir = os.path.expandvars(data_dir)
        self.shard_paths = sorted(glob.glob(os.path.join(expanded_dir, "*.npy")))
        first_shard = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = first_shard.shape[0]
        last_shard = np.load(self.shard_paths[-1], mmap_mode='r')
        self.full_len = ((len(self.shard_paths) - 1) * self.shard_size) + last_shard.shape[0]
        self.len = self.full_len if limit == 0 else min(self.full_len, limit)
        self.cache_size = cache_size
        self.memmaps = OrderedDict()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} out of range")
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        if shard_idx not in self.memmaps:
            if len(self.memmaps) >= self.cache_size:
                self.memmaps.popitem(last=False)
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
        else:
            self.memmaps.move_to_end(shard_idx)
        # return torch.from_numpy(np.array(self.memmaps[shard_idx][local_idx])).bfloat16()
        return torch.from_numpy(np.array(self.memmaps[shard_idx][local_idx])).float()

class LabeledMemmapDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, norm_fn=None, cache_size=100, limit=0, balance=True):
        self.pos_paths = sorted(glob.glob(os.path.join(os.path.expandvars(pos_dir), "*.npy")))
        self.neg_paths = sorted(glob.glob(os.path.join(os.path.expandvars(neg_dir), "*.npy")))
        
        def get_stats(p):
            if not p: return 0, 0
            sz = np.load(p[0], mmap_mode='r').shape[0]
            return (len(p) - 1) * sz + np.load(p[-1], mmap_mode='r').shape[0], sz
        if balance:
            n_pos_shards, n_neg_shards = len(self.pos_paths), len(self.neg_paths)
            self.pos_paths = self.pos_paths[:min(n_pos_shards, n_neg_shards)]
            self.neg_paths = self.neg_paths[:min(n_pos_shards, n_neg_shards)]
            
        pos_full, self.pos_sz = get_stats(self.pos_paths)
        neg_full, self.neg_sz = get_stats(self.neg_paths)
        
        if limit > 0:
            self.pos_len = min(pos_full, limit // 2 + max(0, limit - limit // 2 - neg_full))
            self.neg_len = min(neg_full, limit - self.pos_len)
            self.pos_len = min(pos_full, limit - self.neg_len)
        else:
            self.pos_len, self.neg_len = pos_full, neg_full
            
        self.len = self.pos_len + self.neg_len
        self.cache_size = cache_size
        self.norm_fn = norm_fn
        self.memmaps = None 
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not 0 <= idx < self.len: 
            raise IndexError(idx)
            
        if self.memmaps is None:
            self.memmaps = OrderedDict()

        is_pos = idx < self.pos_len
        paths, sz = (self.pos_paths, self.pos_sz) if is_pos else (self.neg_paths, self.neg_sz)
        shard_idx, local_idx = divmod(idx if is_pos else idx - self.pos_len, sz)
        cache_key = (is_pos, shard_idx)
        
        if cache_key not in self.memmaps:
            self.cache_misses += 1
            if len(self.memmaps) >= self.cache_size: 
                self.memmaps.popitem(last=False)
            self.memmaps[cache_key] = np.load(paths[shard_idx], mmap_mode='r')
        else:
            self.cache_hits += 1
            self.memmaps.move_to_end(cache_key)
            
        x = torch.from_numpy(np.array(self.memmaps[cache_key][local_idx])).float()
        y = torch.tensor(1.0 if is_pos else 0.0, dtype=torch.float32)
        
        if self.norm_fn:
            x = self.norm_fn(x)
        return x, y

class TissueMemmapDataset(Dataset):
    """Tissue-classification dataset over uint8 sharded memmaps.

    Companion to ``scripts/bam_to_labeled_memmap.py``. The expected
    directory layout is::

        <data_dir>/
            schema.json
            manifest.parquet      # one row per output window
            shard_NNNNN.npy       # (<=shard_size, context, n_features) uint8

    The manifest is the single source of truth for labels and split logic.
    Each row carries ``(shard_idx, row_idx, read_name, tissue_str,
    cell_str, tissue_id, cell_id, crop_start, read_length)``. This class
    indexes by manifest row and loads the corresponding shard slice on
    demand. Shards stay on disk as raw uint8; the cast to float32 happens
    per item, and any normalization is applied online by the caller via
    ``norm_fn``.

    Train / val splitting is a polars expression evaluated against the
    manifest at construction time::

        from smrt_foundation.dataset import TissueMemmapDataset

        # held-out cell: leave out cell_id 0 from training
        train = TissueMemmapDataset(out_dir, filter_expr=pl.col('cell_id') != 0)
        val   = TissueMemmapDataset(out_dir, filter_expr=pl.col('cell_id') == 0)

        # random read-level split using a precomputed mask column
        m = pl.read_parquet(out_dir + '/manifest.parquet').with_columns(
            pl.lit(np.random.RandomState(0).rand(...)).alias('rand'))
        train = TissueMemmapDataset(out_dir, filter_expr=pl.col('rand') >= 0.2)

    Args:
        data_dir: Directory containing schema.json, manifest.parquet, and
            shard_*.npy files.
        filter_expr: Optional polars expression applied to the manifest at
            init time to restrict the dataset (for train/val splitting,
            tissue subsetting, etc.). Default: include all rows.
        norm_fn: Optional callable applied to the float32 window before
            return (e.g. ``normalize_read_mad`` from
            ``smrt_foundation.normalization``). Default: no normalization.
        cache_size: Max number of shard memmaps held open at once. LRU.
            Each shard is a few hundred MB at the default
            ``shard_size=16384`` and ``context=4096``, so a small cache
            keeps memory bounded; OS page cache handles the actual data
            caching at finer granularity.

    ``__getitem__(idx)`` returns ``(x, tissue_id)`` where ``x`` is a
    float32 tensor of shape ``(context, n_features)`` and ``tissue_id`` is
    a long tensor scalar. Tissue id assignments are recorded in
    ``schema.json``'s ``tissue_to_id`` field.
    """

    def __init__(self, data_dir, filter_expr=None, norm_fn=None, cache_size=8):
        self.data_dir = os.path.expandvars(data_dir)

        manifest_path = os.path.join(self.data_dir, 'manifest.parquet')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.parquet missing at {manifest_path}. "
                f"Did the bam_to_labeled_memmap job complete?"
            )
        manifest = pl.read_parquet(manifest_path)
        if filter_expr is not None:
            manifest = manifest.filter(filter_expr)
        if manifest.height == 0:
            raise ValueError(
                f"filter_expr eliminated every row of {manifest_path}. "
                f"Check that the expression matches at least one read."
            )

        # Three columns are all we need at __getitem__ time. Stored as a
        # 2D ndarray for O(1) indexing (faster than polars row-by-row).
        self.refs = manifest.select(
            ['shard_idx', 'row_idx', 'tissue_id']
        ).to_numpy().astype(np.int64, copy=False)

        self.cache_size = cache_size
        self.norm_fn = norm_fn
        self.memmaps = OrderedDict()

    def __len__(self):
        return len(self.refs)

    def _shard_path(self, shard_idx):
        return os.path.join(self.data_dir, f"shard_{int(shard_idx):05d}.npy")

    def _get_shard(self, shard_idx):
        if shard_idx in self.memmaps:
            self.memmaps.move_to_end(shard_idx)
            return self.memmaps[shard_idx]
        if len(self.memmaps) >= self.cache_size:
            self.memmaps.popitem(last=False)
        arr = np.load(self._shard_path(shard_idx), mmap_mode='r')
        self.memmaps[shard_idx] = arr
        return arr

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.refs):
            raise IndexError(idx)
        shard_idx, row_idx, tissue_id = self.refs[idx]
        shard = self._get_shard(int(shard_idx))
        x = torch.from_numpy(np.array(shard[int(row_idx)])).float()
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        y = torch.tensor(int(tissue_id), dtype=torch.long)
        return x, y


def compute_log_normalization_stats(df, features, epsilon=1):
    def clean(col):
        return df[col].explode().fill_null(0).cast(pl.Float32).clip(lower_bound=0).log1p()
    means = {col: clean(col).mean() for col in features}
    stds = {col: clean(col).std() for col in features}
    return means, stds

class PairedGapMemmapDataset(Dataset):
    """Frozen SSL pair validation set with known gap distances.

    Loads positive pairs from a directory written by
    `scripts/build_ssl_pair_val.py`. Each pair is a (view1, view2) tuple
    of non-overlapping `target_len`-base windows from the same molecule,
    separated by a known number of real (non-pad) bases. The integer gap
    label is exposed alongside the views so the eval loop can compute
    per-gap top-k accuracy / cosine-sim correlation without any extra
    bookkeeping.

    Directory layout (mirrors the build script):
        <data_dir>/
            pairs/
                shard_00000.npy    # shape (shard_size, 2, T, n_features) float16
                shard_00001.npy
                ...
            gaps.npy                # shape (N_total,) int32
            metadata.yaml           # gaps used, target_len, source, etc.

    Args:
        data_dir: Path to the val set directory.
        norm_fn: Optional normalization callable (e.g. KineticsNorm). When
            provided it is applied independently to view1 and view2 — same
            semantics as `LabeledMemmapDataset.norm_fn`.
        cache_size: Max number of shard memmaps held open at once. LRU.
        gap_filter: Optional iterable of gap_bp values to restrict the
            dataset to (useful for evaluating one gap at a time without
            loading the rest). When None, all pairs are exposed.
        limit: If > 0, cap the total number of pairs returned.

    `__getitem__(idx)` returns `(view1, view2, gap_bp)` where view1 and
    view2 are float tensors of shape `(target_len, n_features)` and
    `gap_bp` is a Python int.
    """

    def __init__(self, data_dir, norm_fn=None, cache_size=100, gap_filter=None, limit=0):
        self.data_dir = os.path.expandvars(data_dir)
        self.pairs_dir = os.path.join(self.data_dir, "pairs")
        self.shard_paths = sorted(glob.glob(os.path.join(self.pairs_dir, "*.npy")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No pair shards found in {self.pairs_dir}")

        # Load gap sidecar.
        gaps_path = os.path.join(self.data_dir, "gaps.npy")
        if not os.path.exists(gaps_path):
            raise FileNotFoundError(f"Missing gaps sidecar at {gaps_path}")
        self.gaps_all = np.load(gaps_path)

        # Compute shard sizes from the actual files (last shard may be partial).
        first_shard = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = int(first_shard.shape[0])
        last_shard = np.load(self.shard_paths[-1], mmap_mode='r')
        full_len = (len(self.shard_paths) - 1) * self.shard_size + int(last_shard.shape[0])
        if full_len != len(self.gaps_all):
            raise ValueError(
                f"Pair count mismatch: {full_len} pairs in shards but "
                f"{len(self.gaps_all)} entries in gaps.npy. Build script "
                f"or directory has been corrupted."
            )

        # Apply gap filter (and optional limit) to derive the index list
        # this dataset actually exposes. Original (unfiltered) shard
        # offsets stay in self.gaps_all so the cache key arithmetic is
        # straightforward.
        if gap_filter is not None:
            mask = np.isin(self.gaps_all, np.asarray(list(gap_filter), dtype=np.int32))
            self.indices = np.nonzero(mask)[0]
        else:
            self.indices = np.arange(full_len, dtype=np.int64)
        if limit and limit > 0:
            self.indices = self.indices[:limit]

        self.cache_size = cache_size
        self.norm_fn = norm_fn
        self.memmaps = OrderedDict()

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range")
        true_idx = int(self.indices[idx])
        shard_idx, local_idx = divmod(true_idx, self.shard_size)
        if shard_idx not in self.memmaps:
            if len(self.memmaps) >= self.cache_size:
                self.memmaps.popitem(last=False)
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
        else:
            self.memmaps.move_to_end(shard_idx)
        pair = np.array(self.memmaps[shard_idx][local_idx])  # (2, T, C)
        v1 = torch.from_numpy(pair[0]).float()
        v2 = torch.from_numpy(pair[1]).float()
        if self.norm_fn is not None:
            v1 = self.norm_fn(v1)
            v2 = self.norm_fn(v2)
        return v1, v2, int(self.gaps_all[true_idx])

    @property
    def gaps(self):
        """Per-sample gap labels for the *exposed* (post-filter) indices.

        Convenience for downstream batch-stratified evaluators that need
        to know each sample's gap without calling `__getitem__`.
        """
        return self.gaps_all[self.indices]


class PairedViewDataset(Dataset):
    """Wrap any Dataset that yields a single tensor and return (view1, view2).

    For SimCLR-style contrastive pretraining. On each __getitem__ call, the
    inner dataset is read once (one disk/memmap hit), then `policy(x)` is
    invoked, returning a pair of independently-augmented views of the same
    sample. Each pair shares the same underlying read — that shared origin
    is what makes them a positive in the contrastive task.

    The inner dataset is expected to return either a bare tensor `[T, C]`
    or a tuple whose first element is such a tensor; if a labelled dataset
    is passed by mistake the label is dropped (contrastive SSL discards
    labels during pretraining).

    The `policy` object must be callable with a single tensor argument and
    must return a `(view1, view2)` tuple. See
    `smrt_foundation.augment.AugmentationPolicy` for the canonical
    implementation.

    Usage:
        inner = ShardedMemmapDataset(ob007_memmap_path, limit=200_000)
        policy = AugmentationPolicy(target_len=32, ...)
        ds = PairedViewDataset(inner, policy=policy, norm_fn=ssl_norm)
        dl = DataLoader(ds, batch_size=512, num_workers=8, pin_memory=True)

    If `norm_fn` is provided it is applied to the raw sample *before* the
    augmentation policy. That matches the convention in ssl_26 / ssl_29
    (KineticsNorm applied dataset-side) and means augmentations operate in
    z-scored space, where a kinetic value of 0 = per-channel mean.
    """

    def __init__(self, inner: Dataset, policy, norm_fn=None):
        self.inner = inner
        self.policy = policy
        self.norm_fn = norm_fn

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x = self.inner[idx]
        if isinstance(x, (tuple, list)):
            x = x[0]
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        v1, v2 = self.policy(x)
        return v1, v2


class NormedDataset(Dataset):
    """Thin wrapper that applies a normalization callable to each sample,
    plus an optional random crop down to `crop_len` along the time axis.

    Expects `inner[idx]` to return a Tensor (not a `(x, y)` tuple). When
    `crop_len` is set and the sample is longer along axis 0, a random
    contiguous window of length `crop_len` is taken — this is how
    short-context SSL runs train on native-length (~4096 bp) reads
    without the model's PositionalEncoding buffer hitting a shape
    mismatch. ssl_29 set the precedent ("random cropping from 4096->128").
    """

    def __init__(self, inner, norm_fn, crop_len=None):
        self.inner = inner
        self.norm_fn = norm_fn
        self.crop_len = crop_len

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        sample = self.norm_fn(self.inner[idx])
        if self.crop_len is not None and sample.shape[0] > self.crop_len:
            start = int(torch.randint(0, sample.shape[0] - self.crop_len + 1, (1,)).item())
            sample = sample[start:start + self.crop_len]
        return sample


class LegacyMethylDataset(IterableDataset):
    def __init__(self, data_path, means, stds, context, restrict_row_groups=100, single_strand=False, inference=False, norm=True):
        super().__init__()
        self.data_path = Path(data_path)
        self.means, self.stds = means, stds
        self.context = context
        self.single_strand = single_strand
        self.inference = inference
        self.restrict = restrict_row_groups
        self.norm = norm

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
        seq_arr = np.stack(df['seq'].str.split("").list.eval(pl.element().replace_strict(self.vocab, default=4)).to_numpy())
        seq_t = torch.tensor(seq_arr, dtype=torch.long)

        kin_list = []
        for k in self.kin_feats:
            if self.norm:
                raw = np.nan_to_num(df[k].to_numpy(), nan=0.0).astype(np.float32)
                np.clip(raw, 0, None, out=raw)
                vals = (np.log1p(raw) - self.means[k]) / (self.stds[k] + 0.1)
            else:
                vals = df[k].to_numpy()
            kin_list.append(vals)
        kin_t = torch.tensor(np.stack(kin_list, axis=1), dtype=torch.float)

        mask = torch.zeros((seq_t.shape[0], seq_t.shape[1], 1), dtype=torch.float)
        labels = torch.tensor(df['label'].to_numpy(), dtype=torch.long) if not self.inference else None

        if self.inference:
            r_names, pos = df['read_name'].to_list(), df['cg_pos'].to_list()

        fwd_data = torch.cat([seq_t.unsqueeze(-1).to(torch.float), kin_t[:, 0:2].permute(0, 2, 1), mask], dim=2)

        rev_data = None
        if self.single_strand:
            rev_seq_t = torch.flip(self.comp_map[seq_t], dims=[1])
            rev_kin = torch.flip(kin_t[:, 2:4], dims=[2]).permute(0, 2, 1)
            rev_data = torch.cat([rev_seq_t.unsqueeze(-1).to(torch.float), rev_kin, mask], dim=2)

        for i in range(len(df)):
            item_fwd = {'data': fwd_data[i].clone()}
            if labels is not None: item_fwd['label'] = labels[i]
            
            if self.inference:
                item_fwd['metadata'] = {'read_name': r_names[i], 'position': pos[i], 'strand': 'fwd' if self.single_strand else 'ds'}
            yield item_fwd

            if rev_data is not None:
                item_rev = {'data': rev_data[i].clone()}
                if labels is not None: item_rev['label'] = labels[i]
                
                if self.inference:
                    item_rev['metadata'] = {'read_name': r_names[i], 'position': pos[i], 'strand': 'rev'}
                yield item_rev
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