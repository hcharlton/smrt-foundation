import os
import gwf
from gwf import Workflow
from gwf import AnonymousTarget

CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/smrt-foundation',
    'gdk_account': 'mutationalscanning',
    'model_config_path': 'smrt-foundation/config.yaml',
    'da1_data':{
        'bam': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
        'denomination': 'da1',
        'ds': 'data/01_processed/ssl_sets/da1.parquet',
        'optional_tags': ['np'],
        'n_reads': 1000,
        'context': 2048
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



def create_ssl_dataset(bam_path, output_path, n_reads, context, optional_tags, denomination, config_dict):
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
        --config_dict {config_dict} \\
        --denomination {denomination}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


da1_data_target = gwf.target_from_template(
    name="create_da1_ssl_dataset",
    template=create_ssl_dataset(
        bam_path=CONFIG['da1_data']['bam'],
        output_path=CONFIG['da1_data']['ds'],
        n_reads=CONFIG['da1_data']['n_reads'],
        context=CONFIG['da1_data']['context'],
        denomination=CONFIG['da1_data']['denomination'],
        optional_tags=CONFIG['da1_data']['optional_tags'],
        config_dict=CONFIG['model_config_path']
    )
)