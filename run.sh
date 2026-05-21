#!/usr/bin/env bash
# Run an experiment. Auto-detects environment (local / Gefion / GenomeDK).
# Resource specs are read from the experiment's config.yaml `resources:` section.
# Job output goes to the experiment directory as <jobid>.out.
#
# Usage:
#   bash run.sh scripts/experiments/ssl_21_pretrain
#   bash run.sh scripts/experiments/supervised_20_full_v2 --mem=512gb

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash run.sh <experiment_directory> [sbatch overrides...]"
    exit 1
fi

EXP_DIR="$1"; shift
SCRIPT="${EXP_DIR}/train.py"
CONFIG="${EXP_DIR}/config.yaml"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read a resource value from config.yaml, with a default fallback.
# Auto-quotes `walltime:` values so PyYAML (YAML 1.1) can't read an unquoted
# HH:MM:SS as a base-60 int (e.g. 15:00:00 → 54000, which sbatch then reads
# as 54000 minutes = 37.5 days).
read_resource() {
    python3 - "$CONFIG" "$1" "$2" <<'PYEOF'
import re, sys, yaml
config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path) as f:
    text = f.read()
text = re.sub(
    r'(?m)^(\s*walltime:[ \t]+)(?!["\'])([^\s#]+)',
    r'\1"\2"',
    text,
)
c = yaml.safe_load(text)
r = c.get('resources', {})
print(r.get(key, default))
PYEOF
}

CORES=$(read_resource cores 16)
MEMORY=$(read_resource memory 256gb)
WALLTIME=$(read_resource walltime 24:00:00)
GRES=$(read_resource gres gpu:8)
NUM_PROCS=$(read_resource num_processes 8)
PARTITION=$(read_resource partition '')

# Read a value from the optional `stage:` block in config.yaml.
# Used on GenomeDK to copy the SSL memmap to local NVMe ($TMPDIR) before
# launching training. Ignored on Gefion / local.
read_stage() {
    python3 - "$CONFIG" "$1" "$2" <<'PYEOF'
import sys, yaml
config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path) as f:
    c = yaml.safe_load(f)
print((c.get('stage') or {}).get(key, default))
PYEOF
}

STAGE_ENABLED=$(read_stage enabled "false")
STAGE_N_SHARDS=$(read_stage n_shards "0")
STAGE_SHUFFLE=$(read_stage shuffle "false")
STAGE_SEED=$(read_stage seed "42")
STAGE_PARALLEL=$(read_stage parallel "16")
SSL_DATASET=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('ssl_dataset','yoran_raw.memmap'))")

# Multi-source datasets (ssl_61+): config can declare `ssl_datasets:` as a
# dict of {source_name: path}. When present, run.sh stages every path under
# $TMPDIR (preserving each basename) and exports SMRT_SSL_MEMMAP_DIR to the
# parent ($TMPDIR), so the training script can resolve each source as
# $SMRT_SSL_MEMMAP_DIR/<basename>. Single-source `ssl_dataset:` continues
# to work unchanged when `ssl_datasets:` is absent.
HAS_SSL_DATASETS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print('1' if c.get('ssl_datasets') else '0')")
SSL_DATASETS_PATHS=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
for v in (c.get('ssl_datasets') or {}).values():
    print(v)
")

# Partition-aware GPU flag selection. When `partition` is set in the config's
# resources block (GenomeDK convention, e.g. `gpu-h200`), submit with
# `--partition=$PARTITION --gpus=$NUM_PROCS` per the cluster's documented
# request style. When unset (legacy / Gefion / earlier GenomeDK partitions),
# fall through to `--gres=$GRES` so existing experiment configs keep working
# without modification.
if [ -n "$PARTITION" ]; then
    GPU_FLAGS="--partition=${PARTITION} --nodes=1 --ntasks=1 --gpus-per-node=${NUM_PROCS}"
else
    GPU_FLAGS="--gres=${GRES}"
fi

# Environment detection
if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ENV="gefion"
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ENV="genomedk"
else
    ENV="local"
fi

echo "[$ENV] $SCRIPT (config: $CONFIG)"
echo "  cores=$CORES mem=$MEMORY time=$WALLTIME procs=$NUM_PROCS gpu_flags=${GPU_FLAGS}"

# Build the GenomeDK staging prefix. On other environments STAGE_CMD is a
# no-op (`true`). $TMPDIR is intentionally NOT expanded here — it must be
# resolved inside the SLURM allocation, where the per-job /tmp/<jobid> exists.
STAGE_CMD="true"
if [ "$ENV" = "genomedk" ] && { [ "$STAGE_ENABLED" = "true" ] || [ "$STAGE_ENABLED" = "True" ]; }; then
    SHUFFLE_FLAG=""
    if [ "$STAGE_SHUFFLE" = "true" ] || [ "$STAGE_SHUFFLE" = "True" ]; then
        SHUFFLE_FLAG="--shuffle"
    fi
    if [ "$HAS_SSL_DATASETS" = "1" ]; then
        STAGE_CMD=""
        while IFS= read -r DS_PATH; do
            [ -z "$DS_PATH" ] && continue
            DS_BASENAME=$(basename "$DS_PATH")
            STAGE_CMD+="bash ${PROJECT_ROOT}/scripts/stage_ssl_to_tmpdir.sh ${PROJECT_ROOT}/${DS_PATH} \$TMPDIR/${DS_BASENAME} --n-shards ${STAGE_N_SHARDS} ${SHUFFLE_FLAG} --seed ${STAGE_SEED} --parallel ${STAGE_PARALLEL} && "
        done <<< "$SSL_DATASETS_PATHS"
        STAGE_CMD+="export SMRT_SSL_MEMMAP_DIR=\$TMPDIR"
        echo "  stage=enabled (multi-source) n_shards=${STAGE_N_SHARDS} shuffle=${STAGE_SHUFFLE} seed=${STAGE_SEED} parallel=${STAGE_PARALLEL}"
        echo "  sources:"
        while IFS= read -r DS_PATH; do
            [ -z "$DS_PATH" ] && continue
            echo "    - $DS_PATH"
        done <<< "$SSL_DATASETS_PATHS"
    else
        STAGE_CMD="bash ${PROJECT_ROOT}/scripts/stage_ssl_to_tmpdir.sh ${PROJECT_ROOT}/data/01_processed/ssl_sets/${SSL_DATASET} \$TMPDIR/${SSL_DATASET} --n-shards ${STAGE_N_SHARDS} ${SHUFFLE_FLAG} --seed ${STAGE_SEED} --parallel ${STAGE_PARALLEL} && export SMRT_SSL_MEMMAP_DIR=\$TMPDIR/${SSL_DATASET}"
        echo "  stage=enabled n_shards=${STAGE_N_SHARDS} shuffle=${STAGE_SHUFFLE} seed=${STAGE_SEED} parallel=${STAGE_PARALLEL}"
    fi
fi

case "$ENV" in
    local)
        echo "WARNING: Running locally (no GPU). Use HPC for real training."
        source "${PROJECT_ROOT}/.venv/bin/activate"
        accelerate launch --num_processes=1 "$SCRIPT" "$CONFIG"
        ;;
    gefion)
        sbatch --job-name="exp_$(basename "$EXP_DIR")" \
               --account=cu_0030 \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               ${GPU_FLAGS} \
               --output="${PROJECT_ROOT}/${EXP_DIR}/%j.out" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && accelerate launch --num_processes=${NUM_PROCS} --mixed_precision=no ${SCRIPT} ${CONFIG}"
        ;;
    genomedk)
        sbatch --job-name="exp_$(basename "$EXP_DIR")" \
               --account=mutationalscanning \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               ${GPU_FLAGS} \
               --output="${PROJECT_ROOT}/${EXP_DIR}/%j.out" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && ${STAGE_CMD} && accelerate launch --num_processes=${NUM_PROCS} --mixed_precision=no ${SCRIPT} ${CONFIG}"
        ;;
esac
