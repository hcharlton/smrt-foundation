"""Baseline dataset-scaling grid (random-init, single-stage). Uses ds_grid."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from scripts.ds_grid import main

if __name__ == "__main__":
    main(sys.argv[1])
