#!/usr/bin/env bash
# Submit each FT recipe under an FT experiment root via run.sh. Forwards any
# --dependency=... flag through to sbatch. Prints the colon-joined Slurm job
# IDs to stdout for use in --dependency=afterok at the next stage.
#
# Usage:
#   bash scripts/ft_grid.sh <ft_exp_root> [--dependency=...] [recipe1 recipe2 ...]
#
# When no recipes are given, iterates over every subdirectory of <ft_exp_root>
# that contains a config.yaml.
#
# Example:
#   FT_DEPS=$(bash scripts/ft_grid.sh \
#       scripts/experiments/supervised_53_finetune_revamp \
#       --dependency=afterok:${REPROBE_DEPS})

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <ft_exp_root> [--dependency=...] [recipe1 ...]" >&2
    exit 1
fi

FT_ROOT="$1"; shift

DEP_FLAG=""
RECIPES=()
for arg in "$@"; do
    case "$arg" in
        --dependency=*) DEP_FLAG="$arg" ;;
        *)              RECIPES+=("$arg") ;;
    esac
done

if [ ! -d "$FT_ROOT" ]; then
    echo "Error: $FT_ROOT is not a directory" >&2
    exit 1
fi

if [ ${#RECIPES[@]} -eq 0 ]; then
    for d in "$FT_ROOT"/*/; do
        [ -f "${d}config.yaml" ] && RECIPES+=("$(basename "$d")")
    done
fi

if [ ${#RECIPES[@]} -eq 0 ]; then
    echo "Error: no recipe subdirs with config.yaml under $FT_ROOT" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

JOBIDS=()
for recipe in "${RECIPES[@]}"; do
    recipe_dir="${FT_ROOT}/${recipe}"
    if [ ! -f "${recipe_dir}/config.yaml" ]; then
        echo "  skip ${recipe}: no config.yaml" >&2
        continue
    fi
    out=$(bash "${PROJECT_ROOT}/run.sh" "$recipe_dir" $DEP_FLAG)
    jobid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}')
    if [ -z "$jobid" ]; then
        echo "Error: failed to extract jobid for $recipe; run.sh output:" >&2
        echo "$out" >&2
        exit 1
    fi
    echo "  ft ${recipe} -> ${jobid}" >&2
    JOBIDS+=("$jobid")
done

if [ ${#JOBIDS[@]} -eq 0 ]; then
    echo "Error: no jobs submitted" >&2
    exit 1
fi

(IFS=:; echo "${JOBIDS[*]}")
