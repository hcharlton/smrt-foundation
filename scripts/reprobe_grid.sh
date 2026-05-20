#!/usr/bin/env bash
# Submit one reprobe sbatch job per size_* subdir under an SSL experiment root.
# Prints the colon-joined Slurm job IDs to stdout for use in --dependency=afterok.
# Per-job progress lines go to stderr.
#
# Usage:
#   bash scripts/reprobe_grid.sh <ssl_exp_root> [size1 size2 ...]
#
# When no sizes are given, discovers every size_* subdir under <ssl_exp_root>.
# Pass an explicit list to subset the grid (e.g. for a smoke test).
#
# Example:
#   REPROBE_DEPS=$(bash scripts/reprobe_grid.sh \
#       scripts/experiments/ssl_58_autoencoder_grid)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <ssl_exp_root> [size1 size2 ...]" >&2
    exit 1
fi

SSL_ROOT="$1"; shift
SIZES=("$@")

if [ ! -d "$SSL_ROOT" ]; then
    echo "Error: $SSL_ROOT is not a directory" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ "$PROJECT_ROOT" == /dcai/* ]]; then
    ACCOUNT=cu_0030
elif [[ "$PROJECT_ROOT" == /home/* ]]; then
    ACCOUNT=mutationalscanning
else
    echo "Error: reprobe_grid.sh requires sbatch; run on Gefion or GenomeDK" >&2
    exit 1
fi

if [ ${#SIZES[@]} -eq 0 ]; then
    for d in "$SSL_ROOT"/size_*; do
        [ -d "$d" ] || continue
        SIZES+=("$(basename "$d" | sed 's/^size_//')")
    done
fi

if [ ${#SIZES[@]} -eq 0 ]; then
    echo "Error: no size_* subdirs under $SSL_ROOT" >&2
    exit 1
fi

JOBIDS=()
for size in "${SIZES[@]}"; do
    exp_dir="${SSL_ROOT}/size_${size}"
    if [ ! -d "$exp_dir" ]; then
        echo "  skip ${size}: ${exp_dir} not a directory" >&2
        continue
    fi
    jobid=$(sbatch --parsable \
        --account=${ACCOUNT} \
        --gres=gpu:1 --cpus-per-task=8 --mem=64gb --time=04:00:00 \
        --job-name="reprobe_${size}" \
        --output="${exp_dir}/reprobe_%j.out" \
        --wrap="source ${PROJECT_ROOT}/.venv/bin/activate && cd ${PROJECT_ROOT} && python -m scripts.ft_eval reprobe --exp_dir ${exp_dir}")
    echo "  reprobe ${size} -> ${jobid}" >&2
    JOBIDS+=("$jobid")
done

if [ ${#JOBIDS[@]} -eq 0 ]; then
    echo "Error: no jobs submitted" >&2
    exit 1
fi

(IFS=:; echo "${JOBIDS[*]}")
