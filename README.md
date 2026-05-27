This is how to get the NBFNet working with Blackwell. 

1. From a venv made with Python==3.10.20: 
    - pip install -r requirements.txt 
2. Acquire a CUDO toolkit on PATHT with `nvcc` (>= 12.8 for Blackwell)
    - pip `nvidia-*` have wheels that include runtime libs but not the compiler. 
    - Set CUDA_HOME and add $CUDA_HOME/bin to PATH 

Why: `nvcc` is NVIDIA's CUDA compiler. It turns .cu source files into machine code for the GPU. The CUDA toolkit is the whole bundle around `nvcc`. It's the compiler, GPU-side C/C++ headers (`cuda_runtime.h`, `cusparse.h`). 

TorchDrug doesn't ship its key kernel (`generalized_rspmm`) as a precompiled binary. It only ships the source (`rspmm.cu`, `spmm.cu`) which JIT-compiles. JIT needs nvcc. 

I installed `cuda-12.9` at `/home/hodgesf/cuda-12.0/` 

To tell the shell and torch where it is I wrote a custom run.sh script in the root directory, so I can just run that and not affect my system cuda. 

3. Update CUDA lib headers (`cuSPARSE`/`cuBLAS`/`cuSolver`/`cuda_runtime`/`cuda_nvrtc`) that JIT need. Then point CPATH at them during launch. (they live in rbfnet-venv/lib/python3.10/site-packages/nvidia/*/include)

CPATH is environment variable that C/C++/CUDA compuler reads at compule time. It is a list of directories that get treated as -isystem flags. nvcc compules rspmm.cu and that file has `#include <cusparse.h>`, the compiler walks CPATH looking for it. 

TorchDrug's kernel sources include CUDA-library headers. My local CUDA toolkit ships nvcc but not the library headers. `nvidia-*-cu12` wheels do ship the headers inside the venv. So we need to point CPATH at the venv copies. 

the `run.sh` script already handles this. 

4. 
