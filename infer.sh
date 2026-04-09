#!/usr/bin/env bash
# Run an inference/eval script. Auto-detects environment (local / Gefion / GenomeDK).
# Resource specs are read from the eval directory's config.yaml `resources:` section.
# Job output goes to the eval directory as <jobid>.out.
#
# Mirrors run.sh's dir+config interface but calls plain `python` (no
# `accelerate launch`) since inference is single-GPU and doesn't need DDP.
#
# Usage:
#   bash infer.sh report/eval/supervised/exp31
#   bash infer.sh report/eval/supervised/exp31 --mem=128gb

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash infer.sh <eval_directory> [sbatch overrides...]"
    exit 1
fi

EVAL_DIR="$1"; shift
SCRIPT="${EVAL_DIR}/infer.py"
CONFIG="${EVAL_DIR}/config.yaml"

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

CORES=$(read_resource cores 4)
MEMORY=$(read_resource memory 64gb)
WALLTIME=$(read_resource walltime 02:00:00)
GRES=$(read_resource gres gpu:1)

# Environment detection
if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ENV="gefion"
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ENV="genomedk"
else
    ENV="local"
fi

echo "[$ENV] $SCRIPT (config: $CONFIG)"
echo "  cores=$CORES mem=$MEMORY time=$WALLTIME gres=$GRES"

case "$ENV" in
    local)
        echo "WARNING: Running locally (no GPU). Use HPC for real inference."
        source "${PROJECT_ROOT}/.venv/bin/activate"
        python "$SCRIPT" "$CONFIG"
        ;;
    gefion)
        sbatch --job-name="eval_$(basename "$EVAL_DIR")" \
               --account=cu_0030 \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               --gres="$GRES" \
               --output="${PROJECT_ROOT}/${EVAL_DIR}/%j.out" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && python ${SCRIPT} ${CONFIG}"
        ;;
    genomedk)
        sbatch --job-name="eval_$(basename "$EVAL_DIR")" \
               --account=mutationalscanning \
               --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
               --gres="$GRES" \
               --output="${PROJECT_ROOT}/${EVAL_DIR}/%j.out" \
               "$@" \
               --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate data_prep && cd ${PROJECT_ROOT} && python ${SCRIPT} ${CONFIG}"
        ;;
esac
