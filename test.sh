#!/usr/bin/env bash
# Run tests. Auto-detects environment (local / Gefion / GenomeDK).
# Job output goes to tests/ directory as <jobid>.out.
#
# Usage:
#   bash test.sh tests/test_kinetics_norm.py
#   bash test.sh tests/
#   bash test.sh tests/test_cpg_pipeline_fidelity.py --mem=64gb

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash test.sh <test_path> [sbatch overrides...]"
    exit 1
fi

TEST_PATH="$1"; shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment detection
if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ENV="gefion"
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ENV="genomedk"
else
    ENV="local"
fi

echo "[$ENV] pytest $TEST_PATH"

case "$ENV" in
    local)
        source "${PROJECT_ROOT}/.venv/bin/activate"
        python -m pytest "$TEST_PATH" -v
        ;;
    gefion)
        JOB_NAME="test_$(basename "$TEST_PATH" .py)"
        sbatch --job-name="$JOB_NAME" \
               --account=cu_0030 \
               --cpus-per-task=4 --mem=32gb --time=01:00:00 \
               --output="${PROJECT_ROOT}/tests/%j.out" \
               "$@" \
               --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && python -m pytest ${TEST_PATH} -v --tb=short"
        ;;
    genomedk)
        JOB_NAME="test_$(basename "$TEST_PATH" .py)"
        sbatch --job-name="$JOB_NAME" \
               --account=mutationalscanning \
               --cpus-per-task=4 --mem=32gb --time=01:00:00 \
               --output="${PROJECT_ROOT}/tests/%j.out" \
               "$@" \
               --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate data_prep && cd ${PROJECT_ROOT} && python -m pytest ${TEST_PATH} -v --tb=short"
        ;;
esac
