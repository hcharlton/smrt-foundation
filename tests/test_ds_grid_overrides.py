"""Tests for `scripts/ds_grid.py` size_overrides resolution and validation.

Run:
    python -m pytest tests/test_ds_grid_overrides.py -v
"""

import copy
import os
import sys

import pytest

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from scripts.ds_grid import resolve_size_config, _validate_size_overrides


def make_base_config():
    """Minimal config with the two sections the resolver touches."""
    return {
        'classifier': {
            'd_model': 128,
            'n_layers': 4,
            'n_head': 4,
            'context': 32,
            'weight_decay': 0.02,
            'pct_start': 0.1,
            'batch_size': 4096,
            'bs_floor': 64,
            'bs_k': 8,
            'max_lr': 3e-3,
        },
        'scaling': {
            'train_sizes': [100, 500, 8000],
            'val_limit': 1000000,
            'max_epochs': 200,
            'min_steps': 100,
            'max_steps': 400000,
            'n_evals': 40,
            'first_eval_step': 100,
        },
    }


# ---------------------------------------------------------------------------
# resolve_size_config
# ---------------------------------------------------------------------------

def test_resolve_returns_original_when_no_overrides():
    config = make_base_config()
    resolved = resolve_size_config(config, 8000)
    assert resolved is config


def test_resolve_returns_original_when_size_not_in_overrides():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_lr': 3e-4}}
    resolved = resolve_size_config(config, 8000)
    assert resolved is config


def test_resolve_merges_classifier_key():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_lr': 3e-4}}
    resolved = resolve_size_config(config, 100)
    assert resolved['classifier']['max_lr'] == 3e-4
    # Non-overridden keys preserved
    assert resolved['classifier']['d_model'] == 128
    # Other sections untouched
    assert resolved['scaling']['max_epochs'] == 200


def test_resolve_merges_scaling_key():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_epochs': 50}}
    resolved = resolve_size_config(config, 100)
    assert resolved['scaling']['max_epochs'] == 50
    assert resolved['classifier']['max_lr'] == 3e-3


def test_resolve_merges_keys_from_both_sections():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_lr': 3e-4, 'max_epochs': 50}}
    resolved = resolve_size_config(config, 100)
    assert resolved['classifier']['max_lr'] == 3e-4
    assert resolved['scaling']['max_epochs'] == 50


def test_resolve_does_not_mutate_input():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_lr': 3e-4, 'max_epochs': 50}}
    snapshot = copy.deepcopy(config)
    _ = resolve_size_config(config, 100)
    assert config == snapshot


# ---------------------------------------------------------------------------
# _validate_size_overrides
# ---------------------------------------------------------------------------

def test_validate_noop_without_section():
    config = make_base_config()
    _validate_size_overrides(config)  # should not raise
    assert 'size_overrides' not in config


def test_validate_normalizes_string_int_keys():
    config = make_base_config()
    config['size_overrides'] = {'100': {'max_lr': 3e-4}}
    _validate_size_overrides(config)
    assert 100 in config['size_overrides']
    assert '100' not in config['size_overrides']


def test_validate_rejects_non_int_coercible_key():
    config = make_base_config()
    config['size_overrides'] = {'not_a_size': {'max_lr': 3e-4}}
    with pytest.raises(ValueError, match="must be an integer"):
        _validate_size_overrides(config)


def test_validate_rejects_unknown_override_key():
    config = make_base_config()
    config['size_overrides'] = {100: {'max_lrate': 3e-4}}  # typo
    with pytest.raises(ValueError, match="unknown keys"):
        _validate_size_overrides(config)


def test_validate_rejects_size_not_in_train_sizes():
    config = make_base_config()
    config['size_overrides'] = {99999: {'max_lr': 3e-4}}
    with pytest.raises(ValueError, match="not in scaling.train_sizes"):
        _validate_size_overrides(config)


def test_validate_rejects_classifier_scaling_collision():
    config = make_base_config()
    config['classifier']['max_epochs'] = 999  # synthetic collision
    config['size_overrides'] = {100: {'max_epochs': 50}}
    with pytest.raises(ValueError, match="share keys"):
        _validate_size_overrides(config)
