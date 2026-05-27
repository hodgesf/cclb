#!/usr/bin/env bash

# Launches NBFNet with the CUDA/torch env it needs. Scoped to this script's
# subshell — does not pollute your interactive shell.

  set -e

  VENV=/home/hodgesf/Desktop/code/cclb/nbfnet-venv
  NV=$VENV/lib/python3.10/site-packages/nvidia

  export CUDA_HOME=/home/hodgesf/cuda-12.9
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  export TORCH_CUDA_ARCH_LIST=12.0
  export CPATH=$NV/cusparse/include:$NV/cublas/include:$NV/cusolver/include:$NV/cuda_runtime/include:$NV/cuda_nvrtc/include
  export PYTORCH_ALLOC_CONF=expandable_segments:True

  cd "$(dirname "$0")/NBFNet"
  exec "$VENV/bin/python" -u script/run.py "$@"