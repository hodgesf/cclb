#!/bin/bash
#SBATCH --job-name=cclb-1e8diag
#SBATCH --output=/nfs/stak/users/hodgesf/results/masked_1e8_diag_%j.log
#SBATCH --error=/nfs/stak/users/hodgesf/results/masked_1e8_diag_%j.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100,vram80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

source /nfs/hpc/share/hodgesf/nbfnet-venv/bin/activate
module load cuda/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="9.0"

cd /nfs/stak/users/hodgesf/code/cclb

python NBFNet/script/run.py \
  -c NBFNet/config/knowledge_graph/rdkg_masked_1e8.yaml \
  --gpus '[0]'
