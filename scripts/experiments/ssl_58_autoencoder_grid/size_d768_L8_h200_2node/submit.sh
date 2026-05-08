#!/bin/bash
# Standalone multi-node submission for ssl_58 size_d768_L8_h200_2node.
#
# Why a separate script instead of `bash run.sh ...`:
#   run.sh's GenomeDK branch wraps a single `accelerate launch` inside
#   sbatch's --wrap, which works for single-node DDP but cannot do the
#   multi-node rendezvous Accelerate needs (one accelerate process per
#   node, each with its own --machine_rank, sharing a --main_process_ip).
#   This script issues an `srun` with --ntasks-per-node=1 so each node
#   runs its own Accelerate launcher, and they coordinate via MASTER_ADDR.
#
# Submit:
#   sbatch scripts/experiments/ssl_58_autoencoder_grid/size_d768_L8_h200_2node/submit.sh
#
# Resource and accelerate parameters are duplicated from config.yaml's
# `resources:` block here as #SBATCH headers and accelerate flags. If
# either is changed, update both — config.yaml is documentation in this
# experiment, not the source of truth.

#SBATCH --account=mutationalscanning
#SBATCH --partition=gpu-h200
#SBATCH --job-name=exp_size_d768_L8_h200_2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --time=02:00:00
#SBATCH --output=/faststorage/project/mutationalscanning/Workspaces/chcharlton/smrt-foundation/scripts/experiments/ssl_58_autoencoder_grid/size_d768_L8_h200_2node/%j.out

set -euo pipefail

export PROJECT_ROOT="/faststorage/project/mutationalscanning/Workspaces/chcharlton/smrt-foundation"
export EXP_DIR="$PROJECT_ROOT/scripts/experiments/ssl_58_autoencoder_grid/size_d768_L8_h200_2node"

cd "$PROJECT_ROOT"
source .venv/bin/activate

# Rendezvous: head node = first hostname in the allocation. SLURM_JOB_NODELIST
# is something like `gn-[1003-1004]`; scontrol expands it to per-host names.
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500

# Verbose NCCL output so the sysadmins have something to read when this
# either succeeds or fails on the inter-node fabric.
export NCCL_DEBUG=INFO

echo "[submit.sh] PROJECT_ROOT=$PROJECT_ROOT"
echo "[submit.sh] EXP_DIR=$EXP_DIR"
echo "[submit.sh] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "[submit.sh] SLURM_NNODES=$SLURM_NNODES"
echo "[submit.sh] MASTER_ADDR=$MASTER_ADDR"
echo "[submit.sh] MASTER_PORT=$MASTER_PORT"
echo "[submit.sh] launching accelerate on each node via srun..."

# srun spawns one task per node (--ntasks-per-node=1 from sbatch headers).
# Each task runs accelerate launch, which then forks 4 GPU worker processes
# within the node. Total ranks = 2 nodes x 4 GPUs = 8.
#
# `bash -c '...'` with single quotes: $SLURM_NODEID expands inside the
# inner bash (per-task), giving 0 on the head node and 1 on the worker.
# $MASTER_ADDR / $MASTER_PORT / $EXP_DIR are exported above and
# propagated by srun's default --export=ALL.
srun bash -c '
exec accelerate launch \
    --num_machines=2 \
    --num_processes=8 \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --mixed_precision=no \
    $EXP_DIR/train.py $EXP_DIR/config.yaml
'
