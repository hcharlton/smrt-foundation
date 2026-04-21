"""Entry point for `bash run.sh scripts/experiments/profile_ssl51_compute_bound`.

Delegates to scripts/profile_ssl51_compute_bound.py. The config.yaml path
that run.sh passes as argv[1] is accepted and ignored — profile knobs
(sizes, warmup, timed-step count, memmap path) are hardcoded in the
profile script to keep it lean.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.profile_ssl51_compute_bound import main

if __name__ == '__main__':
    main()
