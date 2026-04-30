"""Thin entry point for the matrixed fine-tune grid; delegates to ds_grid_v2."""
import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.ds_grid_v2 import main

if __name__ == "__main__":
    main(sys.argv[1])
