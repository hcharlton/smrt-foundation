"""
Tests for smrt_foundation.utils.check_resume_compatible.

The function gates Accelerate state-dict loading on architecture match;
silent failure here corrupts a training run, so the branches are tested
directly:

  - missing run_metadata.yaml sidecar raises
  - arch_key mismatch raises and names the offending key
  - git_hash mismatch warns (does not raise) when arch matches
  - all-match returns the stored config

Run:
    python -m pytest tests/test_resume_compatible.py -v
"""

import os
import sys

import pytest
import yaml

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.utils import check_resume_compatible


ARCH_KEYS = ('d_model', 'n_layers', 'n_head', 'context', 'p_mask', 'mask_size')


def _write_sidecar(resume_dir, stored):
    with open(os.path.join(resume_dir, 'run_metadata.yaml'), 'w') as f:
        yaml.dump(stored, f)


def _base_smrt2vec():
    return {
        'd_model': 128, 'n_layers': 4, 'n_head': 4,
        'context': 512, 'p_mask': 0.15, 'mask_size': 10,
    }


def test_missing_sidecar_raises(tmp_path):
    config = {'smrt2vec': _base_smrt2vec()}
    with pytest.raises(RuntimeError, match="run_metadata.yaml"):
        check_resume_compatible(str(tmp_path), config, ARCH_KEYS)


def test_arch_mismatch_raises(tmp_path):
    stored = {'smrt2vec': _base_smrt2vec()}
    _write_sidecar(str(tmp_path), stored)
    config = {'smrt2vec': {**_base_smrt2vec(), 'd_model': 256}}
    with pytest.raises(RuntimeError, match=r"smrt2vec\.d_model differs"):
        check_resume_compatible(str(tmp_path), config, ARCH_KEYS)


def test_mask_key_mismatch_raises(tmp_path):
    stored = {'smrt2vec': _base_smrt2vec()}
    _write_sidecar(str(tmp_path), stored)
    config = {'smrt2vec': {**_base_smrt2vec(), 'p_mask': 0.30}}
    with pytest.raises(RuntimeError, match=r"smrt2vec\.p_mask differs"):
        check_resume_compatible(str(tmp_path), config, ARCH_KEYS)


def test_git_hash_mismatch_warns(tmp_path, capsys):
    stored = {'smrt2vec': _base_smrt2vec(), 'git_hash': 'aaaaaaaaaaaa' + 'a' * 28}
    _write_sidecar(str(tmp_path), stored)
    config = {'smrt2vec': _base_smrt2vec(), 'git_hash': 'bbbbbbbbbbbb' + 'b' * 28}
    result = check_resume_compatible(str(tmp_path), config, ARCH_KEYS)
    captured = capsys.readouterr()
    assert 'WARNING: git hash differs' in captured.out
    assert result['git_hash'] == stored['git_hash']


def test_all_match_returns_stored(tmp_path, capsys):
    stored = {'smrt2vec': _base_smrt2vec(), 'git_hash': 'cafebabe' * 5}
    _write_sidecar(str(tmp_path), stored)
    config = {'smrt2vec': _base_smrt2vec(), 'git_hash': 'cafebabe' * 5}
    result = check_resume_compatible(str(tmp_path), config, ARCH_KEYS)
    captured = capsys.readouterr()
    assert 'WARNING' not in captured.out
    assert result['smrt2vec'] == stored['smrt2vec']


def test_arch_keys_subset_ignores_other_diffs(tmp_path):
    """Only the keys in arch_keys are compared; other smrt2vec entries can
    differ freely (e.g. batch_size, lr) without blocking resume."""
    stored = {'smrt2vec': {**_base_smrt2vec(), 'batch_size': 512, 'max_lr': 3e-4}}
    _write_sidecar(str(tmp_path), stored)
    config = {'smrt2vec': {**_base_smrt2vec(), 'batch_size': 1024, 'max_lr': 1e-3}}
    result = check_resume_compatible(str(tmp_path), config, ARCH_KEYS)
    assert result['smrt2vec']['d_model'] == 128
