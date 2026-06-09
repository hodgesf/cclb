#!/bin/bash
#SBATCH --job-name=cclb-audit
#SBATCH --output=/nfs/stak/users/hodgesf/results/audit_%j.log
#SBATCH --error=/nfs/stak/users/hodgesf/results/audit_%j.err
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

# === EDIT THIS ===
# Pick the checkpoint to audit. Example: best checkpoint from masked_1e8_10ep.
CKPT="/nfs/stak/users/hodgesf/experiments/KnowledgeGraphCompletion/RDKG/MaskedNBFNet/2026-06-07-XX-XX-XX/model_epoch_7.pth"
CFG="NBFNet/config/knowledge_graph/rdkg_masked_1e8.yaml"
# ==================

# Run all four audits sequentially against the same checkpoint.
# Each evaluates on valid + test. With fast_test=1000 each takes ~5 min.
# Without fast_test, each takes ~70 min, so all four total ~5 hours.

for AUDIT in baseline shuffle random permute; do
    echo ""
    echo "=========================================="
    echo "  Running audit: $AUDIT"
    echo "  $(date)"
    echo "=========================================="
    python NBFNet/script/audit.py \
        -c $CFG \
        --checkpoint $CKPT \
        --audit $AUDIT \
        --gpus '[0]'
done

echo ""
echo "All audits done at $(date)"
