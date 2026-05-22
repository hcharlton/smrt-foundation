"""Thin entry point: delegate to the v3 grid trainer."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.ds_grid_v3 import main

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    main(config_path)
