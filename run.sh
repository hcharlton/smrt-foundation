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

# Read a resource value from config.yaml, with a default fallback
read_resource() {
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
r = c.get('resources', {})
print(r.get('$1', '$2'))
"
}

CORES=$(read_resource cores 16)
MEMORY=$(read_resource memory 256gb)
WALLTIME=$(read_resource walltime 24:00:00)
GRES=$(read_resource gres gpu:8)
NUM_PROCS=$(read_resource num_processes 8)

# Environment detection
if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ENV="gefion"
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ENV="genomedk"
else
    ENV="local"
fi

echo "[$ENV] $SCRIPT (config: $CONFIG)"
echo "  cores=$CORES mem=$MEMORY time=$WALLTIME gres=$GRES procs=$NUM_PROCS"

case "$ENV" in
    local)
        echo "WARNING: Running locally (no GPU). Use HPC for real training."
        source "${PROJECT_ROOT}/.venv/bin/activate"
        accelerate launch --num_processes=1 "$SCRIPT" "$CONFIG"
        ;;
    gefion)
        sbatch --job-name="exp_$(basename "$EXP_DIR")" \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               --gres="$GRES" \
               --output="${PROJECT_ROOT}/${EXP_DIR}/%j.out" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && accelerate launch --num_processes=${NUM_PROCS} --mixed_precision=no ${SCRIPT} ${CONFIG}"
        ;;
    genomedk)
        sbatch --job-name="exp_$(basename "$EXP_DIR")" \
               --account=mutationalscanning \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               --gres="$GRES" \
               --output="${PROJECT_ROOT}/${EXP_DIR}/%j.out" \
               "$@" \
               --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate data_prep && cd ${PROJECT_ROOT} && accelerate launch --num_processes=${NUM_PROCS} --mixed_precision=no ${SCRIPT} ${CONFIG}"
        ;;
esac
