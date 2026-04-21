"""Entry point for `bash run.sh scripts/experiments/profile_ssl51_dataloader_grid`.

Delegates to scripts/profile_ssl51_dataloader_grid.py. Config path passed
as argv[1] is accepted and ignored — grid and sizes are hardcoded in the
profile script.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.profile_ssl51_dataloader_grid import main

if __name__ == '__main__':
    main()
