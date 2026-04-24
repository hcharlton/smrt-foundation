"""Fine-tune ssl_53 d256_L8 SimCLR encoder with per-size schedule overrides."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from scripts.ds_grid import main

if __name__ == "__main__":
    main(sys.argv[1])
