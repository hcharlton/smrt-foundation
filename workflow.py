import os
from gwf import Workflow, AnonymousTarget


############################# ENVIRONMENT ######################################

curr_dir = os.getcwd()
IS_GEFION = curr_dir.startswith('/dcai')

if IS_GEFION:
    root_dir = '/dcai/projects/cu_0030/smrt-foundation'
    gwf_defaults = {}
else:
    root_dir = '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/smrt-foundation'
    gwf_defaults = {'account': 'mutationalscanning'}


############################# CONFIG ###########################################

# Add more dataset entries here and they'll get processed automatically.
CONFIG = {
    'project_root': root_dir,
    'data_config': 'configs/data.yaml',
    'stats': {
        'path': 'data/02_analysis/norm_stats.yaml',
        'num_threads': 12,
        'chunk_stride': 5,
        'idx_stride': 20,
    },
    'datasets': {
        # not on gefion yet
        # 'da1': {
        #     'bam': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
        #     'zarr': 'data/01_processed/ssl_sets/da1.zarr',
        #     'memmap': 'data/01_processed/ssl_sets/da1.memmap',
        #     'optional_tags': [],
        #     'n_reads': 0,
        # },
        'ob007': {
            'bam': 'data/00_raw/unlabeled/ob007_kinetics_diploid.bam',
            'zarr': 'data/01_processed/ssl_sets/ob007.zarr',
            'memmap': 'data/01_processed/ssl_sets/ob007.memmap',
            'memmap_raw':  'data/01_processed/ssl_sets/ob007_raw.memmap',
            'memmap_filter_qual':  'data/01_processed/ssl_sets/ob007_filter_qual.memmap',
            'optional_tags': ['sm', 'sx'],
            'n_reads': 0,
        },
        'yoran': {
            'bam': 'data/00_raw/unlabeled/yoran_kinetics_diploid.bam',
            'zarr': 'data/01_processed/ssl_sets/yoran.zarr',
            'memmap': 'data/01_processed/ssl_sets/yoran.memmap',
            'memmap_raw':  'data/01_processed/ssl_sets/yoran_raw.memmap',
            'memmap_filter_qual':  'data/01_processed/ssl_sets/yoran_filter_qual.memmap',
            'optional_tags': ['sm', 'sx'],
            'n_reads': 0,
        },
        'cpg_pos':{
            'bam': 'data/00_raw/labeled/methylated_hifi_reads.bam',
            'zarr': 'data/01_processed/ssl_sets/cpg_pos.zarr',
            'memmap': 'data/01_processed/val_sets/cpg_pos_v2.memmap',
            'optional_tags': [],
            'n_reads': 0,
        },
        'cpg_neg':{
            'bam': 'data/00_raw/labeled/unmethylated_hifi_reads.bam',
            'zarr': 'data/01_processed/ssl_sets/cpg_neg.zarr',
            'memmap': 'data/01_processed/val_sets/cpg_neg_v2.memmap',
            'optional_tags': [],
            'n_reads': 0,
        },
        'cpg_pos_subset':{
            'bam': 'data/00_raw/labeled/methylated_subset.bam',
            'zarr': 'data/01_processed/ssl_sets/cpg_pos_subset.zarr',
            'memmap': 'data/01_processed/val_sets/cpg_pos_subset.memmap',
            'optional_tags': [],
            'n_reads': 0,
        },
        'cpg_neg_subset':{
            'bam': 'data/00_raw/labeled/unmethylated_subset.bam',
            'zarr': 'data/01_processed/ssl_sets/cpg_neg_subset.zarr',
            'memmap': 'data/01_processed/val_sets/cpg_neg_subset.memmap',
            'optional_tags': [],
            'n_reads': 0,
        }
    }
}

# workflow initialization
gwf = Workflow(defaults=gwf_defaults)

def p(path):
    return os.path.join(CONFIG['project_root'], path)


############################# TEMPLATES ########################################

# --- utility ---

def mock_file(output_path):
    """
    Creates an empty file with an old timestamp (Year 2000).
    Used on Gefion to satisfy BAM dependencies without real data.
    """
    inputs = {}
    outputs = {'out_file': output_path}
    options = {'cores': 1, 'memory': '1gb', 'walltime': '00:10:00'}

    spec = f"""
    mkdir -p $(dirname {output_path})
    touch -t 200001010000 {output_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


# --- data pipeline ---

def zarr_conversion(bam_path, output_path, n_reads, optional_tags, config, profile=False):
    inputs = {'in_file': bam_path}
    outputs = {'out_file': output_path}
    if IS_GEFION:
        options = {'cores': 1, 'memory': '4gb', 'walltime': '00:10:00'}
        spec = f"""
        if [ ! e "{output_path}" ]; then
            echo "ERROR: the zarr file for this job does not appear to already have been transferred to Gefion"
            exit 1
        fi
        echo "updating timestamp of transferred zarr file"
        touch {output_path}
        """
        return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

    tags_str = ' '.join(optional_tags)
    profiler_env = "TimeLINE_PROFILE=1" if profile else ""

    options = {'cores': 12, 'memory': '128gb', 'walltime': '18:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate data_prep
    cd {p('')}
    {profiler_env} python -m scripts.bam_to_zarr \
        --input {bam_path} \
        --n_reads {n_reads} \
        --output_path {output_path} \
        --optional_tags {tags_str} \
        --config {config}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def memmap_conversion(
    zarr_path,
    output_path,
    config_path,
    shard_size=16384,
    shards=0,
    context=4096,
    fwd_features=['seq', 'fi', 'fp'],
    rev_features=['seq', 'ri', 'rp'],
    reverse_complement=True,
    normalize=False,
    filter_qual=False,
    profile=False
):
    inputs = {'in_file': zarr_path}
    outputs = {'out_file': output_path}

    options = {'cores': 8, 'memory': '64gb', 'walltime': '18:00:00'}

    profiler_env = "TimeLINE_PROFILE=1" if profile else ""
    rc_flag = "--reverse_complement" if reverse_complement else ""
    norm_flag = "--normalize" if normalize else ""
    filter_qual_flag = "--filter_qual" if filter_qual else ""
    fwd_str = " ".join(fwd_features)
    rev_str = " ".join(rev_features)

    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate data_prep
    cd {p('')}

    {profiler_env} python -m scripts.zarr_to_memmap_instanceNorm \
        --input_path {zarr_path} \
        --output_path {output_path} \
        --config_path {config_path} \
        --shard_size {shard_size} \
        --max_shards {shards} \
        --context {context} \
        --fwd_features {fwd_str} \
        --rev_features {rev_str} \
        {rc_flag} \
        {norm_flag} \
        {filter_qual_flag}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def memmap_cpg_conversion(
    zarr_path, output_path, config_path,
    shard_size=4*2**20, shards=0, context=32,
    val_pct=0.2, seed=42, profile=False
):
    inputs = {'in_file': zarr_path}
    outputs = {'out_file': f'{output_path}'}
    options = {'cores': 8, 'memory': '64gb', 'walltime': '18:00:00'}

    profiler_env = "TimeLINE_PROFILE=1" if profile else ""
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate data_prep
    cd {p('')}

    {profiler_env} python -m scripts.zarr_to_methyl_memmap_v2 \
        --input_path {zarr_path} \
        --output_path {output_path} \
        --config_path {config_path} \
        --shard_size {shard_size} \
        --max_shards {shards} \
        --context {context} \
        --val_pct {val_pct} \
        --seed {seed}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


# --- validation ---

def validate_memmap(memmap_path, config_path):
    inputs = {'in_file': memmap_path}
    outputs = {'out_file': os.path.join(memmap_path, 'validation.log')}

    options = {'cores': 4, 'memory': '32gb', 'walltime': '00:30:00'}

    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate data_prep
    cd {p('')}

    python -m scripts.validate_memmap_instanceNorm \
        --input_path {memmap_path}\
        --config_path {config_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def inject_norm_stats(zarr_path, chunk_stride, idx_stride, num_threads):
    sentinel = f"{zarr_path}.stats_added"
    inputs = {'infile': zarr_path}
    outputs = {'sentinel': sentinel}
    options = {'cores': num_threads, 'memory': '256gb', 'walltime': '02:00:00'}

    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate data_prep
    cd {p('')}
    python -m scripts.inject_norm_stats \
        --input_path {zarr_path} \
        --chunk_stride {chunk_stride} \
        --idx_stride {idx_stride} \
        --num_threads {num_threads}
    touch {sentinel}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


############################# PIPELINE LOGIC ###################################

def process_dataset(name, data):
    if IS_GEFION:
        gwf.target_from_template(
            name=f"mock_bam_{name}",
            template=mock_file(output_path=data['bam'])
        )

    zarr_target = gwf.target_from_template(
        name=f"{name}_to_zarr",
        template=zarr_conversion(
            bam_path=data['bam'],
            output_path=data['zarr'],
            n_reads=data['n_reads'],
            optional_tags=data['optional_tags'],
            config=CONFIG['data_config'],
            profile=True
        )
    )

    if name.startswith('cpg'):
        gwf.target_from_template(
            name=f'{name}_to_memmap',
            template=memmap_cpg_conversion(
                zarr_path=zarr_target.outputs['out_file'],
                output_path=data['memmap'],
                config_path=CONFIG['data_config'],
            )
        )
        return

    memmap_target = gwf.target_from_template(
        name=f'{name}_to_memmap',
        template=memmap_conversion(
            zarr_path=zarr_target.outputs['out_file'],
            output_path=data['memmap'],
            config_path=CONFIG['data_config'],
            profile=True,
            normalize=True
        )
    )

    memmap_target_raw = gwf.target_from_template(
        name=f'{name}_to_memmap_raw',
        template=memmap_conversion(
            zarr_path=zarr_target.outputs['out_file'],
            output_path=data['memmap_raw'],
            config_path=CONFIG['data_config'],
            profile=True,
            normalize=False,
            shards=50
        )
    )

    memmap_target_filtered = gwf.target_from_template(
        name=f'{name}_to_memmap_filter_qual',
        template=memmap_conversion(
            zarr_path=zarr_target.outputs['out_file'],
            output_path=data['memmap_filter_qual'],
            config_path=CONFIG['data_config'],
            profile=True,
            normalize=False,
            filter_qual=True,
            shards=500
        )
    )

    gwf.target_from_template(
        name=f'{name}_validation',
        template=validate_memmap(
            memmap_path=memmap_target.outputs['out_file'],
            config_path=CONFIG['data_config']
        )
    )

    if name == 'ob007':
        gwf.target_from_template(
            name='stats_test',
            template=inject_norm_stats(
                zarr_path=zarr_target.outputs['out_file'],
                chunk_stride=CONFIG['stats']['chunk_stride'],
                idx_stride=CONFIG['stats']['idx_stride'],
                num_threads=CONFIG['stats']['num_threads'],
            )
        )

        test_memmap_target = gwf.target_from_template(
            name='zarr_to_memmap_test',
            template=memmap_conversion(
                zarr_path=zarr_target.outputs['out_file'],
                output_path='data/01_processed/ssl_sets/ob007_test.memmap',
                config_path=CONFIG['data_config'],
                shards=30,
                context=4096,
                shard_size=16384,
                fwd_features=['seq', 'fi', 'fp'],
                rev_features=['seq', 'ri', 'rp'],
                reverse_complement=True,
                profile=True,
                normalize=True
            )
        )

        gwf.target_from_template(
            name='validate_memmap_test',
            template=validate_memmap(
                memmap_path=test_memmap_target.outputs['out_file'],
                config_path=CONFIG['data_config']
            )
        )


############################# TARGETS ##########################################

for name, data in CONFIG['datasets'].items():
    process_dataset(name, data)
