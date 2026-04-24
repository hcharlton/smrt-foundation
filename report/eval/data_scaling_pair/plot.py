"""
Entry point for the data-scaling trio (scaling, residuals, trajectories)
from a baseline supervised CSV and a finetuned CSV. See
report/eval/_scaling_pair.py for the shared driver.
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
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    main(config_path, args.output_path)
