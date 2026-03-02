import yaml, os, sys
import polars as pl
import altair as alt
from torch.utils.data import DataLoader

module_path = os.path.abspath("/dcai/users/chache/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)


def main():
    chart.save('./compare_feature_dists.svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')
    main()