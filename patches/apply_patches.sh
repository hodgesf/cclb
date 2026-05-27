#!/usr/bin/env bash
# Apply CCLB-specific patches to the torchdrug install in the venv.
#
# These four patches are required for the NBFNet code to run on PyTorch 2.9
# with a Blackwell-class GPU (sm_120). They are not in upstream torchdrug
# 0.2.1 (the last pip-published version, which pins python<3.11) because that
# release predates PyTorch 2.5+ API changes.
#
# Patches applied:
#   1. torchdrug/utils/torch.py
#      cpp_extension positional argument drift -- torch 2.9 inserted
#      extra_sycl_cflags into the signature, so we use keyword args.
#
#   2. torchdrug/layers/functional/extension/spmm.h
#   3. torchdrug/layers/functional/extension/rspmm.h
#      ATen header move: ATen/SparseTensorUtils.h was relocated to
#      ATen/native/SparseTensorUtils.h in newer torch versions.
#
#   4. torchdrug/core/engine.py
#      (a) torch.load default flipped to weights_only=True in torch 2.6;
#          our checkpoints contain torchdrug Graph objects, so we pass
#          weights_only=False explicitly.
#      (b) Drop non-tensor entries (the Graph buffers for graph/fact_graph)
#          from the state_dict before load_state_dict, since PyTorch 2.x
#          expects every state_dict value to be a tensor.
#
# Usage:
#   ./patches/apply_patches.sh                # patches ./nbfnet-venv
#   ./patches/apply_patches.sh /path/to/venv  # patches a venv at another path

set -e

VENV=${1:-./nbfnet-venv}
SITE=$VENV/lib/python3.10/site-packages
TD=$SITE/torchdrug

if [ ! -d "$TD" ]; then
    echo "ERROR: torchdrug not found at $TD" >&2
    echo "Run 'pip install -r NBFNet/requirements.txt' inside the venv first." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Patching torchdrug at $TD"

cp "$SCRIPT_DIR/torchdrug/core/engine.py"                      "$TD/core/engine.py"
cp "$SCRIPT_DIR/torchdrug/utils/torch.py"                      "$TD/utils/torch.py"
cp "$SCRIPT_DIR/torchdrug/layers/functional/extension/spmm.h"  "$TD/layers/functional/extension/spmm.h"
cp "$SCRIPT_DIR/torchdrug/layers/functional/extension/rspmm.h" "$TD/layers/functional/extension/rspmm.h"

echo "Done. Patched files:"
echo "  core/engine.py                       (weights_only + non-tensor state_dict filter)"
echo "  utils/torch.py                       (cpp_extension kwarg compat)"
echo "  layers/functional/extension/spmm.h   (ATen header path)"
echo "  layers/functional/extension/rspmm.h  (ATen header path)"
