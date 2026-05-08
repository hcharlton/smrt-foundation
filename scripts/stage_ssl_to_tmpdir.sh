#!/usr/bin/env bash
# Stage SSL .npy shards from a source memmap dir to a destination dir.
# Designed for HPC nodes with local NVMe (e.g. GenomeDK $TMPDIR), so the
# hot training stream reads from local disk instead of /faststorage.
#
# Usage:
#   bash scripts/stage_ssl_to_tmpdir.sh <src_dir> <dest_dir> \
#       [--n-shards N] [--shuffle] [--seed S] [--parallel P]
#
# Defaults: copy all shard_*.npy, deterministic order, --parallel 16.
# When --n-shards is set: take the first N (sorted) by default, or a
# reproducible random N with --shuffle (seeded by --seed).

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <src_dir> <dest_dir> [--n-shards N] [--shuffle] [--seed S] [--parallel P]" >&2
    exit 1
fi

SRC="$1"; shift
DEST="$1"; shift

N_SHARDS=0
SHUFFLE=0
SEED="42"
PARALLEL=16

while [ $# -gt 0 ]; do
    case "$1" in
        --n-shards)  N_SHARDS="$2"; shift 2 ;;
        --shuffle)   SHUFFLE=1; shift ;;
        --seed)      SEED="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$SRC" ]; then
    echo "[stage] ERROR: src_dir does not exist: $SRC" >&2
    exit 1
fi

TOTAL_AVAIL=$(find "$SRC" -maxdepth 1 -name 'shard_*.npy' | wc -l)
if [ "$TOTAL_AVAIL" -eq 0 ]; then
    echo "[stage] ERROR: no shard_*.npy files found in $SRC" >&2
    exit 1
fi

if [ "$N_SHARDS" -gt 0 ] && [ "$N_SHARDS" -gt "$TOTAL_AVAIL" ]; then
    echo "[stage] WARNING: requested $N_SHARDS but only $TOTAL_AVAIL shards available; staging all $TOTAL_AVAIL"
    N_SHARDS=0
fi

mkdir -p "$DEST"

list_shards() {
    if [ "$N_SHARDS" -gt 0 ] && [ "$SHUFFLE" -eq 1 ]; then
        find "$SRC" -maxdepth 1 -name 'shard_*.npy' \
          | shuf -n "$N_SHARDS" --random-source=<(yes "stage-seed-$SEED")
    elif [ "$N_SHARDS" -gt 0 ]; then
        find "$SRC" -maxdepth 1 -name 'shard_*.npy' | sort | head -n "$N_SHARDS"
    else
        find "$SRC" -maxdepth 1 -name 'shard_*.npy'
    fi
}

if [ "$N_SHARDS" -gt 0 ]; then
    PLAN_COUNT="$N_SHARDS"
else
    PLAN_COUNT="$TOTAL_AVAIL"
fi
MODE="sorted"
if [ "$SHUFFLE" -eq 1 ] && [ "$N_SHARDS" -gt 0 ]; then
    MODE="random (seed=$SEED)"
fi

echo "[stage] src=$SRC"
echo "[stage] dest=$DEST"
echo "[stage] copying $PLAN_COUNT / $TOTAL_AVAIL shards | mode=$MODE | parallel=$PARALLEL"

START_TS=$(date +%s)
list_shards | xargs -P"$PARALLEL" -I{} cp {} "$DEST/"
END_TS=$(date +%s)

COPIED=$(find "$DEST" -maxdepth 1 -name 'shard_*.npy' | wc -l)
echo "[stage] done: $COPIED files at $DEST in $((END_TS - START_TS))s"

if [ "$COPIED" -ne "$PLAN_COUNT" ]; then
    echo "[stage] ERROR: expected $PLAN_COUNT files, found $COPIED" >&2
    exit 1
fi
