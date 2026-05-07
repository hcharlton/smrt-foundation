"""Shared loading and feature-extraction helpers for the tissue probe sweep.

Each probe / EDA script in this directory imports from here. The goal is for
every script to be runnable in isolation: one materialised numpy array per
split, one set of cheap feature extractors, and a few canonical figure
helpers.

The data path used is `data/01_processed/tissue_sets/yoran_ctx4096/`. The
partition lives at `<data_dir>/partition.csv` and was produced by
`scripts/make_tissue_partition.py`.
"""

import os
import sys
import numpy as np
import polars as pl

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from torch.utils.data import DataLoader, Dataset

from smrt_foundation.dataset import TissueMemmapDataset
from smrt_foundation.normalization import KineticsNorm


DATA_DIR = os.path.join(
    _REPO_ROOT, 'data', '01_processed', 'tissue_sets', 'yoran_ctx4096'
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

TISSUES = ['colon', 'kidney', 'liver', 'lung', 'muscle', 'skin', 'spleen', 'testis']

# Default downsampling caps for probe scripts. The yoran ctx4096 dataset is
# ~37 GB on disk; full materialisation per probe is wasteful for diagnostics.
# Each probe script reads these defaults but can override by passing explicit
# `limit` values to `_shared.load_split`. 10k train / 2k val keeps every probe
# under ~1 GB of materialised float32 and well under a minute on a node with
# warm shard IO.
DEFAULT_TRAIN_LIMIT = 10_000
DEFAULT_VAL_LIMIT = 2_000

# Feature column layout in the stored uint8 windows.
SEQ_COL = 0
KIN_COLS = [1, 2, 3, 4]
MASK_COL = 5

# Sequence token map (matches configs/data.yaml).
TOK_A, TOK_C, TOK_G, TOK_T, TOK_N = 0, 1, 2, 3, 4
COMPLEMENT_TOK = np.array([TOK_T, TOK_G, TOK_C, TOK_A, TOK_N], dtype=np.int64)


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_partition(data_dir=DATA_DIR, partition_path=None, strip_whitespace=True):
    """Read partition.csv as a polars DataFrame.

    The on-disk file has a known anomaly: at least one row contains
    `'val_s1 '` (trailing space) for the split column, which would silently
    drop reads from any string-equality filter. By default we strip whitespace
    from the `split` column on load and warn if anything was changed. Pass
    `strip_whitespace=False` to disable and inspect the raw values
    (e.g. `eda_partition_sanity.py` flips this to surface the anomaly).
    """
    partition_path = partition_path or os.path.join(data_dir, 'partition.csv')
    if not os.path.exists(partition_path):
        raise FileNotFoundError(
            f"partition.csv not found at {partition_path}. "
            f"Run scripts/make_tissue_partition.py first."
        )
    df = pl.read_csv(partition_path)
    if strip_whitespace:
        before = df['split'].to_list()
        df = df.with_columns(pl.col('split').str.strip_chars())
        after = df['split'].to_list()
        n_changed = sum(1 for a, b in zip(before, after) if a != b)
        if n_changed:
            print(
                f"[load_partition] WARNING: stripped whitespace from {n_changed} "
                f"split labels in {partition_path}",
                flush=True,
            )
    return df


def load_manifest(data_dir=DATA_DIR):
    """Read manifest.parquet as a polars DataFrame."""
    return pl.read_parquet(os.path.join(data_dir, 'manifest.parquet'))


def assert_partition_sane(data_dir=DATA_DIR, expected_per_tissue=None, tol=0.01):
    """Check that the partition is disjoint, stratified, and cell-mapped correctly.

    Raises AssertionError on any deviation. Cheap: runs in <1 s on the 70k-row
    partition. Every probe script calls this at startup so a corrupt partition
    fails loudly rather than silently invalidating results.

    Args:
        data_dir: Path to the dataset directory.
        expected_per_tissue: Optional dict {split_name: int} of expected per-tissue
            counts. None disables stratification check.
        tol: Tolerance for the per-tissue stratification check (relative).
    """
    partition = load_partition(data_dir)
    manifest = load_manifest(data_dir)

    # Disjoint read_names across splits.
    pairs = [('train', 'val_s1'), ('train', 'val_s2'), ('val_s1', 'val_s2')]
    for a, b in pairs:
        a_set = set(partition.filter(pl.col('split') == a)['read_name'].to_list())
        b_set = set(partition.filter(pl.col('split') == b)['read_name'].to_list())
        overlap = a_set & b_set
        assert not overlap, f"{a} and {b} overlap on {len(overlap)} read_names"

    # Cell mapping check: train + val_s1 from one cell, val_s2 from the other.
    # We don't hardcode which cell_id is which because schema.json maps
    # m84108_250930_153107_s2 -> 0 and m84108_251007_115244_s1 -> 1.
    by_split_cell = (
        partition.join(
            manifest.select(['read_name', 'cell_id']).unique(subset='read_name'),
            on='read_name', how='left'
        )
        .group_by(['split', 'cell_id'])
        .agg(pl.len().alias('n'))
    )
    for split in ('train', 'val_s1'):
        cells = by_split_cell.filter(pl.col('split') == split)
        assert cells.height == 1, f"{split} spans {cells.height} cells: {cells}"
    s2 = by_split_cell.filter(pl.col('split') == 'val_s2')
    assert s2.height == 1, f"val_s2 spans {s2.height} cells: {s2}"
    s1_cell = by_split_cell.filter(pl.col('split') == 'train')['cell_id'][0]
    s2_cell = s2['cell_id'][0]
    assert s1_cell != s2_cell, f"train and val_s2 share cell_id {s1_cell}"

    # Per-tissue stratification check.
    if expected_per_tissue is not None:
        by_split_tissue = (
            partition.join(
                manifest.select(['read_name', 'tissue_id']).unique(subset='read_name'),
                on='read_name', how='left'
            )
            .group_by(['split', 'tissue_id'])
            .agg(pl.len().alias('n'))
        )
        for split, expected in expected_per_tissue.items():
            counts = by_split_tissue.filter(pl.col('split') == split).sort('tissue_id')
            assert counts.height == 8, f"{split} has {counts.height} tissues, want 8"
            ns = counts['n'].to_numpy().astype(np.int64)
            rel = np.abs(ns - expected) / expected
            assert (rel <= tol).all(), (
                f"{split} per-tissue counts {ns.tolist()} deviate >{tol*100:.1f}% from {expected}"
            )

    return partition, manifest


def compute_norm(data_dir=DATA_DIR, max_samples=4_096, n_continuous=4):
    """Compute KineticsNorm stats on a 16k sample of the train split.

    Matches the setup in `_shared_train.py:355-360` so the probes see the
    same per-channel scaling as the deep model.
    """
    partition = load_partition(data_dir)
    train_names = partition.filter(pl.col('split') == 'train')['read_name']
    train_raw = TissueMemmapDataset(
        data_dir, filter_expr=pl.col('read_name').is_in(train_names),
    )
    return KineticsNorm(
        train_raw, log_transform=True, max_samples=max_samples,
        n_continuous=n_continuous,
    )


class _NormCropDataset(Dataset):
    """Wrap a tissue dataset with norm + deterministic centre-crop per item.

    Mirrors `CroppedNormedDataset` in `_shared_train.py` so we can hand it
    off to a multi-worker DataLoader for fast parallel materialisation.
    """
    __slots__ = ('inner', 'norm_fn', 'crop_len')

    def __init__(self, inner, norm_fn, crop_len):
        self.inner = inner
        self.norm_fn = norm_fn
        self.crop_len = int(crop_len)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x, y = self.inner[idx]
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        T = x.shape[0]
        if T > self.crop_len:
            start = (T - self.crop_len) // 2
            x = x[start:start + self.crop_len]
        return x, y


def _materialize(raw_ds, norm_fn, crop_len, n_features=6, num_workers=8,
                 batch_size=256):
    """Iterate raw_ds once with a multi-worker DataLoader and stack results.

    Returns (out, y) with `out` a contiguous (N, crop_len, n_features) float32
    numpy array and `y` an (N,) int64 array of tissue ids. With 16 cores
    available, num_workers=8 saturates faststorage IO without thrashing the
    page cache.
    """
    wrapped = _NormCropDataset(raw_ds, norm_fn, int(crop_len))
    N = len(wrapped)
    out = np.empty((N, int(crop_len), n_features), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)
    dl = DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )
    i = 0
    for x_batch, y_batch in dl:
        b = x_batch.shape[0]
        out[i:i + b] = x_batch.numpy()
        y[i:i + b] = y_batch.numpy()
        i += b
    assert i == N, f"materialised {i} samples but dataset has {N}"
    return out, y


def load_split(
    split,
    norm_fn=None,
    context=2048,
    limit=None,
    data_dir=DATA_DIR,
    verbose=True,
):
    """Load and materialise one of {train, val_s1, val_s2}.

    Returns a dict with numpy arrays:
        X            : (N, context, 6) float32, kinetics + seq + mask
        tissue_id    : (N,) int64
        cell_id      : (N,) int64
        read_length  : (N,) int64
        read_name    : (N,) object (str)

    `norm_fn` is applied if not None (the standard KineticsNorm). `context`
    is the centre-crop length (defaults to 2048 to match the deep model).
    Pass `context=4096` and `norm_fn=None` to get the raw uint8-cast data
    over the full window.
    """
    partition = load_partition(data_dir)
    manifest = load_manifest(data_dir)

    split_names = partition.filter(pl.col('split') == split)['read_name'].to_list()
    if len(split_names) == 0:
        raise ValueError(f"split={split!r} has no rows in partition")

    manifest_split = manifest.filter(pl.col('read_name').is_in(split_names))
    if limit is not None and limit > 0 and manifest_split.height > limit:
        manifest_split = manifest_split.head(limit)

    raw = TissueMemmapDataset(
        data_dir,
        filter_expr=pl.col('read_name').is_in(split_names),
    )
    if limit is not None and limit > 0 and len(raw) > limit:
        raw.refs = raw.refs[:limit]

    if verbose:
        print(f"[{split}] materialising {len(raw)} samples (ctx={context}, "
              f"normed={norm_fn is not None}) ...", flush=True)

    X, y = _materialize(raw, norm_fn, crop_len=int(context))

    # Sanity: tissue_id from the dataset must match the manifest.
    tissue_id = manifest_split['tissue_id'].to_numpy().astype(np.int64)
    cell_id = manifest_split['cell_id'].to_numpy().astype(np.int64)
    read_length = manifest_split['read_length'].to_numpy().astype(np.int64)
    read_name = manifest_split['read_name'].to_numpy()

    assert np.array_equal(y, tissue_id), (
        f"[{split}] dataset tissue_ids do not match manifest tissue_ids "
        f"(first mismatch at index {int(np.argmax(y != tissue_id))})"
    )

    return {
        'X': X,
        'tissue_id': tissue_id,
        'cell_id': cell_id,
        'read_length': read_length,
        'read_name': read_name,
    }


# --- Feature extractors ---------------------------------------------------

def pool_summary(X):
    """Pooled summary features: 25-d.

    For each kinetics channel `{fi, fp, ri, rp}`: [mean, std, p10, p50, p90]
    over the time axis, ignoring positions where mask=1. Plus base-frequency
    on the seq channel `{A, C, G, T, N}`. All in normalised space.
    """
    assert X.ndim == 3 and X.shape[2] == 6, f"expected (N, T, 6), got {X.shape}"
    N = X.shape[0]
    kin = X[..., KIN_COLS]  # (N, T, 4)

    # The yoran build dropped reads shorter than the context, so mask is
    # all-real by construction. We compute per-channel stats over time.
    out = np.empty((N, 4 * 5 + 5), dtype=np.float32)
    for ci in range(4):
        ch = kin[..., ci]
        out[:, ci * 5 + 0] = ch.mean(axis=1)
        out[:, ci * 5 + 1] = ch.std(axis=1)
        # numpy quantile is faster on float32 than torch on cpu for this size.
        qs = np.quantile(ch, [0.10, 0.50, 0.90], axis=1)
        out[:, ci * 5 + 2] = qs[0]
        out[:, ci * 5 + 3] = qs[1]
        out[:, ci * 5 + 4] = qs[2]

    seq = X[..., SEQ_COL].astype(np.int64)
    base_block = out[:, 4 * 5:]
    for tok in (TOK_A, TOK_C, TOK_G, TOK_T, TOK_N):
        base_block[:, tok] = (seq == tok).mean(axis=1)
    return out


def bin_summary(X, n_bins=16):
    """Binned summary features: (n_bins * 4 * 2)-d.

    Divide the time axis into n_bins equal segments. Per segment, per kinetics
    channel: [mean, std].
    """
    assert X.ndim == 3 and X.shape[2] == 6
    N, T, _ = X.shape
    assert T % n_bins == 0, f"T={T} not divisible by n_bins={n_bins}"
    bin_size = T // n_bins

    kin = X[..., KIN_COLS]  # (N, T, 4)
    kin_b = kin.reshape(N, n_bins, bin_size, 4)
    means = kin_b.mean(axis=2)  # (N, n_bins, 4)
    stds = kin_b.std(axis=2)
    feats = np.concatenate([means, stds], axis=2)  # (N, n_bins, 8)
    return feats.reshape(N, n_bins * 4 * 2).astype(np.float32, copy=False)


def flatten_kinetics(X):
    """Flattened raw kinetics: (T * 4)-d."""
    return X[..., KIN_COLS].reshape(X.shape[0], -1).astype(np.float32, copy=False)


def seq_composition(X):
    """5-d base-frequency features {A, C, G, T, N}."""
    seq = X[..., SEQ_COL].astype(np.int64)
    N = seq.shape[0]
    out = np.empty((N, 5), dtype=np.float32)
    for tok in (TOK_A, TOK_C, TOK_G, TOK_T, TOK_N):
        out[:, tok] = (seq == tok).mean(axis=1)
    return out


# --- Altair styling -------------------------------------------------------

TISSUE_COLORS = [
    '#4c78a8', '#e45756', '#54a24b', '#f58518',
    '#72b7b2', '#eeca3b', '#b279a2', '#ff9da6',
]

CELL_COLORS = ['#4c78a8', '#e45756']


def tissue_color_scale():
    import altair as alt
    return alt.Scale(domain=TISSUES, range=TISSUE_COLORS)


def cell_color_scale():
    import altair as alt
    return alt.Scale(domain=['cell_id=0 (s2)', 'cell_id=1 (s1)'], range=CELL_COLORS)


def cell_label(cell_id):
    """Map int cell_id → 'cell_id=0 (s2)' / 'cell_id=1 (s1)'."""
    return np.where(np.asarray(cell_id) == 0, 'cell_id=0 (s2)', 'cell_id=1 (s1)')


def tissue_label(tissue_id):
    arr = np.asarray(tissue_id, dtype=np.int64)
    return np.array([TISSUES[i] for i in arr], dtype=object)
