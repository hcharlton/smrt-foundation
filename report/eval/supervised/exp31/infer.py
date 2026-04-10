"""
Inference for supervised experiment 31 (DirectClassifier baseline).

Config-driven: the first positional arg is a path to a `config.yaml` following
the same style as the training configs under `scripts/experiments/*`. Use
`bash infer.sh report/eval/supervised/exp31` to launch — this mirrors
`bash run.sh scripts/experiments/...` but skips `accelerate launch` since
inference is single-GPU.

Loads a per-epoch checkpoint from exp 31, runs the classifier on the memmap
shards specified in the config, and streams results to a parquet file with one
row per sample:

    input       — nested list [context, 4] of the normalized tensor the model
                  actually saw (post-KineticsNorm, i.e. log1p + z-score applied
                  to channels [1, 2]).
    label       — 1.0 = methylated, 0.0 = unmethylated (from which pos/neg
                  directory the sample came).
    logit       — raw pre-sigmoid model output (the thing BCEWithLogitsLoss
                  consumes at training time).
    shard_path  — full path to the source `.npy` shard the sample was read
                  from. Parquet dictionary-encodes repeats, so storing the
                  full path is cheap on disk.
    shard_row   — int32 row index within that shard. `np.load(shard_path)[row]`
                  recovers the pre-normalized sample.

Results are written batch-by-batch via PyArrow's ParquetWriter — at no point
does the script hold more than one batch of results in memory. This is
necessary at full scale (~86M val samples × 128 floats/sample = ~44 GB if
accumulated).

The checkpoints live on Gefion, so this script is meant to run there. Locally
it can only be smoke-tested — no data is loaded at import time.

Expected config schema:

    experiment_type: 'eval'
    experiment_name: 'supervised_exp31'
    resources: { cores, memory, walltime, gres }   # consumed by infer.sh
    checkpoint: 'path/to/epoch_XX.pt'
    pos_data:   'path/to/pos.memmap/<split>'
    neg_data:   'path/to/neg.memmap/<split>'
    output:     'path/to/predictions.parquet'
    inference:
        batch_size: 4096
        limit: 0
        num_workers: 4

Extension: the exp 32 variant is a sibling `report/eval/supervised/exp32/infer.py`
that swaps `DirectClassifier` for `DirectClassifierNoTransformer` and drops
the `n_layers`/`n_head` constructor args. Everything else is identical.
"""

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier
from smrt_foundation.normalization import KineticsNorm


REQUIRED_KEYS = ['checkpoint', 'pos_data', 'neg_data', 'output']

DEFAULT_INFERENCE = {
    'batch_size': 4096,
    'limit': 0,
    'num_workers': 4,
}

PARQUET_SCHEMA = pa.schema([
    ('input', pa.list_(pa.list_(pa.float32()))),
    ('label', pa.float32()),
    ('logit', pa.float32()),
    ('shard_path', pa.string()),
    ('shard_row', pa.int32()),
])


def load_model(checkpoint_path, device):
    """Load DirectClassifier + training-time KineticsNorm from a checkpoint.

    Mirrors the instantiation path in
    `scripts/experiments/supervised_31_baseline_clean/train.py:178-183` — same
    constructor args from the saved classifier config so the state dict maps
    cleanly. The KineticsNorm stats are the ones saved alongside the weights
    at training time, reconstructed via `KineticsNorm.load_stats`; this is the
    documented inference entry point and avoids re-sampling statistics from
    the val data (which would drift from training).

    The checkpoint is loaded to CPU and only the model is moved to `device`.
    This keeps `norm_means` / `norm_stds` on CPU so that forked DataLoader
    workers can apply the normalization without touching CUDA (CUDA contexts
    don't survive fork, so any CUDA tensor reached from a worker process
    raises `cudaErrorInitializationError`).
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    c = ckpt['config']['classifier']

    model = DirectClassifier(
        d_model=c['d_model'],
        n_layers=c['n_layers'],
        n_head=c['n_head'],
        max_len=c['context'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    norm_fn = KineticsNorm.load_stats(ckpt)
    return model, norm_fn, ckpt


def batch_provenance(ds, row_offset, batch_size):
    """Compute (shard_path, shard_row) for dataset indices [row_offset, row_offset + batch_size).

    Mirrors `LabeledMemmapDataset.__getitem__:103-105`: positives live at
    indices 0..pos_len-1, negatives at pos_len..len-1. Within each half,
    `divmod(local_index, shard_size)` gives the shard and row.

    Computed per-batch rather than precomputed for the full dataset to avoid
    holding ~86M path references in memory at once.
    """
    paths = []
    rows = np.empty(batch_size, dtype=np.int32)
    for i in range(batch_size):
        idx = row_offset + i
        if idx < ds.pos_len:
            s_idx, l_idx = divmod(idx, ds.pos_sz)
            paths.append(ds.pos_paths[s_idx])
        else:
            s_idx, l_idx = divmod(idx - int(ds.pos_len), ds.neg_sz)
            paths.append(ds.neg_paths[s_idx])
        rows[i] = l_idx
    return paths, rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for key in REQUIRED_KEYS:
        assert key in config, f"Missing required config key: {key}"

    c = DEFAULT_INFERENCE | config.get('inference', {})
    config['inference'] = c

    print(f"Eval: {config.get('experiment_type', 'eval')}/"
          f"{config.get('experiment_name', 'unnamed')}")
    print(f"Config: {config}")

    checkpoint_path = os.path.expandvars(config['checkpoint'])
    output_path = os.path.expandvars(config['output'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model, norm_fn, ckpt = load_model(checkpoint_path, device)
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch")}')
    print(f'Checkpoint metrics: {ckpt.get("metrics")}')
    print(f'Norm means: {norm_fn.means.tolist()}, stds: {norm_fn.stds.tolist()}')
    print(f'CNN receptive field: {model.encoder.cnn.r0} bases')

    ds = LabeledMemmapDataset(
        config['pos_data'], config['neg_data'],
        limit=c['limit'], norm_fn=norm_fn, balance=False,
    )
    print(f'Dataset: {len(ds)} samples ({ds.pos_len} pos, {ds.neg_len} neg)')

    dl = DataLoader(
        ds, batch_size=c['batch_size'],
        num_workers=c['num_workers'], pin_memory=True, shuffle=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA)

    row_offset = 0
    with torch.no_grad():
        for x, y in tqdm(dl, desc='Inference'):
            logits = model(x.to(device))

            bs = x.shape[0]
            b_paths, b_rows = batch_provenance(ds, row_offset, bs)

            batch_table = pa.table({
                'input': x.numpy().astype(np.float32).tolist(),
                'label': pa.array(y.numpy(), type=pa.float32()),
                'logit': pa.array(
                    logits.squeeze(-1).float().cpu().numpy(), type=pa.float32()
                ),
                'shard_path': b_paths,
                'shard_row': pa.array(b_rows, type=pa.int32()),
            }, schema=PARQUET_SCHEMA)
            writer.write_table(batch_table)
            row_offset += bs

    writer.close()
    print(f'Wrote {row_offset} rows to {output_path}')


if __name__ == '__main__':
    main()
