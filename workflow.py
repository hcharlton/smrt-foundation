import os
import gwf
from gwf import Workflow
from gwf import AnonymousTarget

CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/smrt-foundation',
    'gdk_account': 'mutationalscanning',
    'config_path': 'smrt_foundation/config.yaml',
    'da1_data':{
        'bam': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
        'denomination': 'da1',
        'ds': 'data/01_processed/ssl_sets/da1.parquet',
        'optional_tags': ['np'],
        'n_reads': 50000,
        'context': 2048
        },
    'zarr_test':{
        'bam': 'data/00_raw/unlabeled/da1_subset_10k.bam',
        'denomination': 'da1',
        'ds': 'data/01_processed/ssl_sets/da1_subset_10k.zarr',
        'optional_tags': [],
        'n_reads': 0,
        },
    'stats':{
        'path': 'data/02_analysis/norm_stats.yaml'
        }
    }

# SLURM backend gwf worker
gwf = Workflow(defaults={'account': CONFIG['gdk_account']})

# resolve paths helper
def p(path):
    return os.path.join(CONFIG['project_root'], path)

def compute_norm_stats(train_parquet_path, output_json_path):
    """Calculates mean/std from the training data."""
    inputs = {'train_set': train_parquet_path}
    outputs = {'stats_file': output_json_path}
    options = {'cores': 16, 
               'memory': '128gb', 
               'walltime': '00:10:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    python -m scripts.compute_norm_stats \\
        -i {train_parquet_path} \\
        -o {output_json_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def zarr_conversion(bam_path, output_path, n_reads, optional_tags, denomination, config, profile=False):
    inputs = {'in_file': bam_path}
    outputs = {'out_file': output_path}

    tags_str = ' '.join(optional_tags)
    profiler_env = "TimeLINE_PROFILE=1" if profile else ""

    options = {'cores': 4, 'memory': '16gb', 'walltime': '00:30:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    {profiler_env} python -m scripts.bam_to_zarr \\
        --input {bam_path} \\
        --n_reads {n_reads} \\
        --output_path {output_path} \\
        --optional_tags {tags_str} \\
        --config {config} \\
        --denomination {denomination}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)
def create_ssl_dataset(bam_path, output_path, n_reads, context, optional_tags, denomination, config):
    inputs = {'in_file': bam_path}
    outputs = {'out_file': output_path}

    tags_str = ' '.join(optional_tags)

    options = {'cores': 16, 'memory': '128gb', 'walltime': '01:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    python -m scripts.make_ssl_dataset \\
        --input {bam_path} \\
        --n_reads {n_reads} \\
        --context {context} \\
        --output_path {output_path} \\
        --optional_tags {tags_str} \\
        --config {config} \\
        --denomination {denomination}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

### ---------- WORKFLOW GRAPH ------------

da1_data_target = gwf.target_from_template(
    name="create_da1_ssl_dataset",
    template=create_ssl_dataset(
        bam_path=CONFIG['da1_data']['bam'],
        output_path=CONFIG['da1_data']['ds'],
        n_reads=CONFIG['da1_data']['n_reads'],
        context=CONFIG['da1_data']['context'],
        denomination=CONFIG['da1_data']['denomination'],
        optional_tags=CONFIG['da1_data']['optional_tags'],
        config=CONFIG['config_path']
    )
)

zarr_conversion_test = gwf.target_from_template(
    name="da1_subset_to_zarr",
    template=zarr_conversion(
        bam_path=CONFIG['zarr_test']['bam'],
        output_path=CONFIG['zarr_test']['ds'],
        n_reads= 100, #CONFIG['zarr_test']['n_reads'],
        denomination=CONFIG['zarr_test']['denomination'],
        optional_tags=CONFIG['zarr_test']['optional_tags'],
        config=CONFIG['config_path'],
        profile=True
    )
)


# stats_target = gwf.target_from_template(
#     name='compute_stats',
#     template=compute_norm_stats(
#         train_parquet_path=da1_data_target.outputs['train_ds'],
#         output_json_path=p(CONFIG['stats']['path'])
#     )
# )