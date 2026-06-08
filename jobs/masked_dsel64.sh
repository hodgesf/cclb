#!/bin/bash
#SBATCH --job-name=cclb-dsel64
#SBATCH --output=/nfs/stak/users/hodgesf/results/masked_dsel64_%j.log
#SBATCH --error=/nfs/stak/users/hodgesf/results/masked_dsel64_%j.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=30:00:00

source /nfs/hpc/share/hodgesf/nbfnet-venv/bin/activate
module load cuda/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="9.0"

cd /nfs/stak/users/hodgesf/code/cclb

python NBFNet/script/run.py \
  -c NBFNet/config/knowledge_graph/rdkg_masked_dsel64.yaml \
  --gpus '[0]'
