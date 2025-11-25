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
     