import os
import subprocess
import yaml
import torch
import polars as pl
import re

def parse_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            dict = yaml.safe_load(f)
        return dict 

def make_optimizer(optimizer_config, model):
    """instantiates an optimizer based on a config file"""
    name = optimizer_config['name']
    params = optimizer_config.get('params', {})
    OptimizerClass = getattr(torch.optim, name)
    return OptimizerClass(model.parameters(), **params)



def parse_schema(schema_dict):
    """
    Parses a schema dictionary from the config into a dictionary
    of Polars dtypes.
    """
    base_types = {
        "String": pl.String,
        "UInt8": pl.UInt8,
        "UInt16": pl.UInt16,
        "UInt32": pl.UInt32,
        "UInt64": pl.UInt64,
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Boolean": pl.Boolean,
    }

    def parse_type(type_str):
        """Recursively parses a type string into a Polars dtype."""
        type_str = type_str.strip().replace("pl.", "")
        
        # Correctly handle List types
        if type_str.startswith("List(") and type_str.endswith(")"):
            inner_type_str = type_str[5:-1]
            # Recursively call parse_type to get the actual inner Polars type object
            inner_type = parse_type(inner_type_str)
            return pl.List(inner_type)
        
        # Look up the base type
        parsed = base_types.get(type_str)
        if parsed is None:
            raise ValueError(f"Polars type '{type_str}' not found in base_types mapping.")
        return parsed

    polars_schema = {
        col_name: parse_type(type_str)
        for col_name, type_str in schema_dict.items()
    }
    return polars_schema


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


class ProgressState:
    """Tracks training progress across interrupted runs.

    Registered with `accelerator.register_for_checkpointing(...)` so the
    counters are part of the Accelerate state directory and restore
    atomically alongside model / optimizer / scheduler / RNG state.
    """
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.step_in_epoch = 0

    def state_dict(self):
        return {
            'epoch': int(self.epoch),
            'global_step': int(self.global_step),
            'step_in_epoch': int(self.step_in_epoch),
        }

    def load_state_dict(self, sd):
        self.epoch = int(sd.get('epoch', 0))
        self.global_step = int(sd.get('global_step', 0))
        self.step_in_epoch = int(sd.get('step_in_epoch', 0))


def check_resume_compatible(resume_dir, config, arch_keys):
    """Refuse to resume if the stored architecture differs from current config.

    Reads `<resume_dir>/run_metadata.yaml` and compares each key in
    `arch_keys` between `config['smrt2vec']` (current) and the stored
    sidecar. Raises `RuntimeError` on missing sidecar or any arch
    mismatch. Emits a warning (not an error) on git-hash mismatch when
    the architecture itself matches.

    `arch_keys` differs per experiment family (contrastive vs masked vs
    MAE vs supervised), so each `_shared_train.py` passes its own tuple.
    """
    sidecar = os.path.join(resume_dir, 'run_metadata.yaml')
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"Resume target {resume_dir} has no run_metadata.yaml sidecar; "
            f"refusing to resume (cannot verify architecture match)."
        )
    with open(sidecar, 'r') as f:
        stored = yaml.safe_load(f)
    cur = config.get('smrt2vec', {})
    prev = stored.get('smrt2vec', {})
    for k in arch_keys:
        if cur.get(k) != prev.get(k):
            raise RuntimeError(
                f"Refusing to resume: smrt2vec.{k} differs "
                f"(stored={prev.get(k)}, current={cur.get(k)}). Architecture "
                f"must match for Accelerate state_dict to load."
            )
    if stored.get('git_hash') and config.get('git_hash') and stored['git_hash'] != config['git_hash']:
        print(
            f"[resume] WARNING: git hash differs "
            f"(stored={stored['git_hash'][:12]}, current={config['git_hash'][:12]}). "
            f"Architecture matches so resume will proceed."
        )
    return stored
