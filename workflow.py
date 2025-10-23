import os
from gwf import Workflow
from gwf import AnonymousTarget

CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/methyl-jepa',
    'gdk_account': 'mutationalscanning',
    'config_path': 'methyl_jepa/config.yaml',
    'jobs':{
        'da1_data':{
            'bam': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
            'ds': 'data/01_processed/inference_sets/da1.parquet',
            'optional_tags': ['np'],
            'n_reads': 1000,
        }
        },
    }

def compute_norm_stats(train_parquet_path, output_json_path):
    """Calculates mean/std from the training data."""
    inputs = {'train_set': train_parquet_path}
    outputs = {'stats_file': output_json_path}
    options = {'cores': 16, 
               'memory': '128gb', 
               'walltime': '00:10:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.compute_norm_stats \\
        -i {train_parquet_path} \\
        -o {output_json_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)



def create_ssl_dataset(bam_path, output_path, n_reads, context, optional_tags):
    inputs = {'in_file': bam_path}
    outputs = {'out_file': output_path}

    tags_str = ' '.join(optional_tags)

    options = {'cores': 32, 'memory': '512gb', 'walltime': '03:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.make_ssl_dataset \\
        -i {bam_path} \\
        -n {n_reads} \\
        -c {context} \\
        -o {output_path} \\
        -t {tags_str}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


da1_data_target = gwf.target_from_template(
    name="create_da1_ssl_dataset",
    template=create_ssl_dataset(
        bam_path=CONFIG['da1_bam_path'],
        output_path=CONFIG['da1_ssl_ds_path'],
        n_reads=100,
        context=2048,
        optional_tags=CONFIG['da1_optional_tags']
    )
)