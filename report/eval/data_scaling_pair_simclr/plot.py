"""
Data-scaling trio (scaling, residuals, trajectories) for exp 47
(random-init baseline on the ds_grid trainer) vs exp 48 (SimCLR d256_L8
encoder finetuned on the same grid). CSVs live on the cluster only.
"""

import os
import sys
import argparse

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from report.eval._scaling_pair import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    main(config_path, args.output_dir)
