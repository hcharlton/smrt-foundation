"""Smoke test for `report.probe_tissue_yoran._shared`.

Exercises the public surface against the live yoran tissue dataset with a
small limit so it stays under a few seconds. Auto-skips on machines that
don't have the data dir present (mirrors the convention in
`tests/test_kinetics_norm.py`).
"""

import os
import numpy as np
import pytest


pytest.importorskip('polars')


from report.probe_tissue_yoran import _shared


pytestmark = pytest.mark.skipif(
    not os.path.exists(_shared.DATA_DIR),
    reason=f"yoran tissue dataset not present at {_shared.DATA_DIR}",
)


def test_load_partition_strips_whitespace():
    df = _shared.load_partition()
    splits = set(df['split'].unique().to_list())
    # The on-disk file has at least one row with `'val_s1 '` (trailing space)
    # which load_partition() must strip by default.
    assert splits.issubset({'train', 'val_s1', 'val_s2'}), (
        f"unexpected split values after strip: {splits}"
    )


def test_assert_partition_sane():
    partition, manifest = _shared.assert_partition_sane()
    assert partition.height > 0
    assert manifest.height > 0


def test_load_split_returns_aligned_arrays():
    data = _shared.load_split('val_s1', norm_fn=None, context=2048, limit=128)
    X = data['X']
    assert X.dtype == np.float32
    assert X.shape == (128, 2048, 6)
    assert data['tissue_id'].shape == (128,)
    assert data['cell_id'].shape == (128,)
    assert data['read_length'].shape == (128,)
    assert len(data['read_name']) == 128
    # mask is the last channel; for the yoran build it is always zero.
    assert (X[..., _shared.MASK_COL] == 0).all()
    # tissue ids must be in [0, 8)
    assert int(data['tissue_id'].min()) >= 0
    assert int(data['tissue_id'].max()) < len(_shared.TISSUES)


def test_feature_extractors_have_expected_shapes():
    data = _shared.load_split('val_s1', norm_fn=None, context=2048, limit=64)
    X = data['X']
    assert _shared.pool_summary(X).shape == (64, 25)
    assert _shared.bin_summary(X, n_bins=16).shape == (64, 128)
    assert _shared.flatten_kinetics(X).shape == (64, 2048 * 4)
    assert _shared.seq_composition(X).shape == (64, 5)


def test_compute_norm_round_trip():
    norm = _shared.compute_norm(max_samples=512)
    assert norm.n_continuous == 4
    # means/stds vectors cover all 6 channels (indices 1..4 are the kinetics)
    assert norm.means.shape[-1] == 6
    assert norm.stds.shape[-1] == 6
    # Kinetics channels should have non-trivial stats; mask/seq stay at defaults.
    for c in (1, 2, 3, 4):
        assert float(norm.stds[c]) > 0
