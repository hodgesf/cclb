#!/bin/bash
#SBATCH --job-name=cclb-fulleval
#SBATCH --output=/nfs/stak/users/hodgesf/results/full_eval_%j.log
#SBATCH --error=/nfs/stak/users/hodgesf/results/full_eval_%j.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

source /nfs/hpc/share/hodgesf/nbfnet-venv/bin/activate
module load cuda/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="9.0"

cd /nfs/stak/users/hodgesf/code/cclb

# Full evaluation (no fast_test) on the headline masked checkpoint.
# Evaluates the unmodified trained model on all valid + all test queries.
# Use this number as the paper's headline test MRR / MR.

CKPT="/nfs/stak/users/hodgesf/experiments/KnowledgeGraphCompletion/RDKG/MaskedNBFNet/2026-06-07-22-28-37/model_epoch_7.pth"
CFG="NBFNet/config/knowledge_graph/rdkg_masked_full.yaml"

echo "=========================================="
echo "  CCLB FULL EVAL on $CKPT"
echo "  $(date)"
echo "=========================================="

python NBFNet/script/audit.py \
    -c $CFG \
    --checkpoint $CKPT \
    --audit baseline \
    --gpus '[0]'

echo ""
echo "Full eval done at $(date)"
