"""
Tests for smrt_foundation.dataset.BalancedChunkedSampler.

The sampler powers ssl_61's dual-source ConcatDataset: each source is a
contiguous index range in a single ConcatDataset, and the sampler must
emit 1/n_sources of the total per source (oversampling smaller sources
via re-shuffle+cycle) in chunks of `chunk_size` round-robin. A bug here
would silently bias pretraining toward whichever source happens to be
larger.

Run:
    python -m pytest tests/test_balanced_chunked_sampler.py -v
"""

import os
import sys

import torch

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import BalancedChunkedSampler


def _classify(indices, source_lengths):
    """Return per-source counts. Source i owns offset[i] to offset[i+1]."""
    offsets = [0]
    for L in source_lengths:
        offsets.append(offsets[-1] + L)
    counts = [0] * len(source_lengths)
    for idx in indices:
        for i in range(len(source_lengths)):
            if offsets[i] <= idx < offsets[i + 1]:
                counts[i] += 1
                break
    return counts


class TestBalancedChunkedSampler:
    def test_equal_sources_emit_50_50(self):
        sampler = BalancedChunkedSampler([1000, 1000], chunk_size=100)
        out = list(iter(sampler))
        assert len(out) == 2000
        counts = _classify(out, [1000, 1000])
        assert counts == [1000, 1000]

    def test_smaller_source_oversampled_to_quota(self):
        # Total = 1100, n=2, so each source must emit 550.
        # Source A only has 100 unique indices -> cycled ~5.5x.
        sampler = BalancedChunkedSampler([100, 1000], chunk_size=10)
        out = list(iter(sampler))
        assert len(out) == 1100
        counts = _classify(out, [100, 1000])
        assert counts == [550, 550]
        a_indices = [i for i in out if i < 100]
        assert all(0 <= i < 100 for i in a_indices)
        # Cycling proven by pigeonhole: 550 emissions over 100 unique slots.
        assert len(set(a_indices)) <= 100

    def test_chunks_are_single_source(self):
        sampler = BalancedChunkedSampler([1000, 1000], chunk_size=50)
        out = list(iter(sampler))
        # Every contiguous run of `chunk_size` indices must come from the
        # same source. We check by walking the stream in groups of 50.
        for start in range(0, len(out), 50):
            group = out[start:start + 50]
            sources = {0 if idx < 1000 else 1 for idx in group}
            assert len(sources) == 1, f"chunk at {start} mixes sources: {group[:5]}..."

    def test_indices_respect_source_offsets(self):
        sampler = BalancedChunkedSampler([300, 500], chunk_size=25)
        out = list(iter(sampler))
        assert all(0 <= idx < 800 for idx in out)
        a_indices = [i for i in out if i < 300]
        b_indices = [i for i in out if i >= 300]
        assert all(0 <= i < 300 for i in a_indices)
        assert all(300 <= i < 800 for i in b_indices)

    def test_len_equals_total_and_iter_is_repeatable(self):
        sampler = BalancedChunkedSampler([400, 600], chunk_size=20)
        assert len(sampler) == 1000
        first = list(iter(sampler))
        second = list(iter(sampler))
        assert len(first) == 1000
        assert len(second) == 1000

    def test_deterministic_under_manual_seed(self):
        sampler = BalancedChunkedSampler([200, 300], chunk_size=10)
        torch.manual_seed(0)
        a = list(iter(sampler))
        torch.manual_seed(0)
        b = list(iter(sampler))
        assert a == b
