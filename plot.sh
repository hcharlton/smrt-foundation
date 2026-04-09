#!/usr/bin/env bash
# Run a plot script. Auto-detects environment (local / Gefion / GenomeDK).
#
# Usage:
#   bash plot.sh report/eda/model_input_heatmaps
#   bash plot.sh report/eda/fi_vs_ri_distributions --mem=128gb

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash plot.sh <plot_directory> [sbatch overrides...]"
    exit 1
fi

PLOT_DIR="$1"; shift
SCRIPT="${PLOT_DIR}/plot.py"
OUTPUT="${PLOT_DIR}/plot.svg"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment detection
if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ENV="gefion"
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ENV="genomedk"
else
    ENV="local"
fi

echo "[$ENV] $SCRIPT → $OUTPUT"

case "$ENV" in
    local)
        source "${PROJECT_ROOT}/.venv/bin/activate"
        python "$SCRIPT" --output_path "$OUTPUT"
        ;;
    gefion)
        sbatch --job-name="plot_$(basename "$PLOT_DIR")" \
               --account=cu_0030 \
               --cpus-per-task=4 --mem=32gb --time=00:30:00 \
               --output="${PLOT_DIR}/slurm_%j.log" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && python ${SCRIPT} --output_path ${OUTPUT}"
        ;;
    genomedk)
        sbatch --job-name="plot_$(basename "$PLOT_DIR")" \
               --account=mutationalscanning \
               --cpus-per-task=4 --mem=32gb --time=00:30:00 \
               --output="${PLOT_DIR}/slurm_%j.log" \
               "$@" \
               --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate data_prep && cd ${PROJECT_ROOT} && python ${SCRIPT} --output_path ${OUTPUT}"
        ;;
esac
