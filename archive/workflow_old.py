import os
import gwf
from gwf import Workflow, AnonymousTarget
curr_dir = os.getcwd()
IS_GEFION = curr_dir.startswith('/dcai')
if IS_GEFION:
    root_dir = 'dcai/users/chache/smrt-foundation'
    gwf_defaults = {'account': 'mutationalscanning'}
else:
    root_dir = '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/smrt-foundation'
    gwf_defaults = {'account': 'mutationalscanning'}
CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/smrt-foundation',
    'gdk_account': 'mutationalscanning',
    'config_path': 'smrt_foundation/config.yaml',
    'stats':{
        'path': 'data/02_analysis/norm_stats.yaml'
        },
    'da1':{
        'bam': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
        'zarr': 'data/01_processed/ssl_sets/da1.zarr',
        'memmap': 'data/01_processed/ssl_sets/da1.memmap',
        'optional_tags': [],
        'n_reads': 0,
        },
    'ob007':{
        'bam': 'data/00_raw/unlabeled/ob007_kinetics_diploid.bam',
        'zarr': 'data/01_processed/ssl_sets/ob007.zarr',
        'memmap': 'data/01_processed/ssl_sets/ob007.memmap',
        'optional_tags': ['sm','sx'],
        'n_reads': 0,
        },
    'stats': {
        'num_threads': 12,
        'chunk_stride': 5,
        'idx_stride': 20,
        }

    }

# SLURM backend gwf worker
gwf = Workflow(defaults={'account': CONFIG['gdk_account']})

# resolve paths helper
def p(path):
    return os.path.join(CONFIG['project_root'], path)


def zarr_conversion(bam_path, output_path, n_reads, optional_tags, config, profile=False):
    inputs = {'in_file': bam_path}
    outputs = {'out_file': output_path}

    tags_str = ' '.join(optional_tags)
    profiler_env = "TimeLINE_PROFILE=1" if profile else ""

    options = {'cores': 12, 'memory': '128gb', 'walltime': '18:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    {profiler_env} python -m scripts.bam_to_zarr \\
        --input {bam_path} \\
        --n_reads {n_reads} \\
        --output_path {output_path} \\
        --optional_tags {tags_str} \\
        --config {config}
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

def inject_norm_stats(zarr_path, chunk_stride, idx_stride, num_threads):
    sentinel = f"{zarr_path}.stats_added"
    inputs = {'infile': zarr_path}
    outputs = {'sentinel': sentinel}
    options = {'cores': num_threads, 'memory': '256gb', 'walltime': '02:00:00'}

    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate jax-prep
    cd {p('')}
    python -m scripts.inject_norm_stats \\
        --input_path {zarr_path} \\
        --chunk_stride {chunk_stride} \\
        --idx_stride {idx_stride} \\
        --num_threads {num_threads}
    touch {sentinel}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

    
def memmap_conversion(
    zarr_path, 
    output_path, 
    config_path, 
    shard_size=16384, 
    shards=0, 
    seq_len=4096,
    fwd_features=['seq', 'fi', 'fp'],
    rev_features=['seq', 'ri', 'rp'],
    reverse_complement=True,
    profile=False
):
    inputs = {'in_file': zarr_path}
    outputs = {'out_file': output_path}
    
    options = {'cores': 4, 'memory': '32gb', 'walltime': '00:30:00'}

    profiler_env = "TimeLINE_PROFILE=1" if profile else ""
    
    rc_flag = "--reverse_complement" if reverse_complement else ""
    
    fwd_str = " ".join(fwd_features)
    rev_str = " ".join(rev_features)

    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    
    {profiler_env} python -m scripts.zarr_to_memmap \\
        --input_path {zarr_path} \\
        --output_path {output_path} \\
        --config_path {config_path} \\
        --shard_size {shard_size} \\
        --max_shards {shards} \\
        --seq_len {seq_len} \\
        --fwd_features {fwd_str} \\
        --rev_features {rev_str} \\
        {rc_flag}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def validate_memmap(
    memmap_path,
    config_path
):
    inputs = {'in_file': memmap_path}
    outputs = {'out_file': os.path.join(memmap_path, 'validation.log')}

    options = {'cores': 4, 'memory': '32gb', 'walltime': '00:30:00'}
    
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate smrt-foundation
    cd {p('')}
    
    python -m scripts.validate_memmap \\
        --input_path {memmap_path}\\
        --config_path {config_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)



### ---------- WORKFLOW GRAPH ------------


da1_to_zarr = gwf.target_from_template(
    name="da1_to_zarr",
    template=zarr_conversion(
        bam_path=CONFIG['da1']['bam'],
        output_path=CONFIG['da1']['zarr'],
        n_reads= CONFIG['da1']['n_reads'],
        optional_tags=CONFIG['da1']['optional_tags'],
        config=CONFIG['config_path'],
        profile=True
    )
)

ob007_to_zarr = gwf.target_from_template(
    name="ob007_to_zarr",
    template=zarr_conversion(
        bam_path=CONFIG['ob007']['bam'],
        output_path=CONFIG['ob007']['zarr'],
        n_reads= CONFIG['ob007']['n_reads'],
        optional_tags=CONFIG['ob007']['optional_tags'],
        config=CONFIG['config_path'],
        profile=True
    )
)


ob007_to_memmap = gwf.target_from_template(
    name='ob007_to_memmap',
    template=memmap_conversion(
        zarr_path=ob007_to_zarr.outputs['out_file'],
        output_path=CONFIG['ob007']['memmap'],
        config_path=CONFIG['config_path'],
        profile=True
    )
)

stats_test = gwf.target_from_template(
    name='stats_test',
    template=inject_norm_stats(
        zarr_path=ob007_to_zarr.outputs['out_file'],
        chunk_stride=CONFIG['stats']['chunk_stride'],
        idx_stride=CONFIG['stats']['idx_stride'],
        num_threads=CONFIG['stats']['num_threads'],
    )
)


# ob007_stats = gwf.target_from_template(
#     name='inject_norm_stats',
#     template=inject_norm_stats(
#         zarr_path=CONFIG['ob007']['zarr'],
#         chunk_stride=CONFIG['stats']['chunk_stride'],
#         idx_stride=CONFIG['stats']['idx_stride'],
#         num_threads=CONFIG['stats']['num_threads'],
#     )
# )




zarr_to_memmap_test = gwf.target_from_template(
    name='zarr_to_memmap_test',
    template=memmap_conversion(
        zarr_path=ob007_to_zarr.outputs['out_file'],
        output_path='data/01_processed/ssl_sets/ob007_test_memmap.memmap',
        config_path=CONFIG['config_path'],
        shards=10,
        seq_len = 4096,
        shard_size=16384,
        fwd_features=['seq', 'fi', 'fp'],
        rev_features=['seq', 'ri', 'rp'],
        reverse_complement=True,
        profile=True
    )
)

validate_memmap_test = gwf.target_from_template(
    name='validate_memmap_test',
    template=validate_memmap(
        memmap_path = zarr_to_memmap_test.outputs['out_file'],
        config_path = CONFIG['config_path'])
)
