# CCLB v2 — NBFNet Baseline on the Rare Disease Knowledge Graph

Faithful-by-construction gene-disease link prediction on RDKG. This repo
contains the **baseline pipeline**: graph pruning, decoupled
propagation/supervision splits, and NBFNet training/evaluation. The
faithful-by-construction (`f-gate`) variant will be added on top of this
baseline in a follow-up.

The end-to-end pipeline is:

```
raw RDKG (jsonl)
   │
   ▼
[ scripts/prune_graph.py ]   ── drops ontology scaffolding, prediction-level
   │                            edges, GO super-hubs; BFS-reachability filter
   ▼
pruned graph (jsonl)
   │
   ▼
[ scripts/build_splits.py ]  ── pair-level 80/10/10 holdout; emits separate
   │                            propagation graph and supervision set
   ▼
train.txt, train_targets.txt, valid.txt, test.txt
   │
   ▼
[ run.sh -> NBFNet/script/run.py ]   ── NBFNet training + eval via TorchDrug
   │
   ▼
checkpoints under ~/experiments/...
```

## Repository layout

```
cclb/
├── NBFNet/                       NBFNet source (vendored, with our edits)
│   ├── config/knowledge_graph/
│   │   └── rdkg.yaml             config for our RDKG run
│   ├── nbfnet/
│   │   └── dataset.py            RDKG dataset class (decoupled split)
│   ├── script/run.py             entry point
│   └── requirements.txt          pip deps
├── scripts/
│   ├── prune_graph.py            6-step graph pruning pipeline
│   └── build_splits.py           pair-level split + decoupled supervision
├── patches/
│   ├── apply_patches.sh          applies the 4 torchdrug patches below
│   └── torchdrug/                patched source files for torchdrug 0.2.1
│       ├── core/engine.py        torch 2.6+ weights_only, non-tensor filter
│       ├── utils/torch.py        cpp_extension kwarg drift (torch 2.9)
│       └── layers/functional/extension/
│           ├── spmm.h            ATen header path move
│           └── rspmm.h           ATen header path move
├── analyze_rdkg.py               graph profiling (nodes, edges, degrees,
│                                  predicates, gene-disease focus)
├── run.sh                        launches NBFNet with the right env vars
├── requirements.txt              top-level pip deps (same as NBFNet/'s)
└── stats.txt                     reference output of analyze_rdkg.py
```

Excluded from the repo (regenerated locally): `nbfnet-venv/`, `data/`,
`splits/`, model checkpoints (`*.pth`), TorchDrug experiment dirs.

## System requirements

- **Python 3.10.** TorchDrug 0.2.1 — the last pip-published release — pins
  `python<3.11`.
- **CUDA 12.8 or newer** for Blackwell GPUs (sm_120). Older CUDA toolkits
  do not have native kernels for sm_120, and PyTorch wheels built against
  CUDA < 12.8 will fail with "no kernel image is available for execution
  on the device".
- **`nvcc` on `PATH`.** TorchDrug ships its `generalized_rspmm` /
  `generalized_spmm` kernels as `.cu` source files and JIT-compiles them
  on first use, which requires the CUDA compiler. The `nvidia-*-cu12` pip
  wheels ship runtime libs but not the compiler, so you need a real CUDA
  toolkit install somewhere on disk.
- **GPU memory.** Training (dim 16, batch size 4) peaks at ~12 GiB.
- **System RAM.** Full evaluation (`full_batch_eval: yes` against all
  936K entities) currently exceeds 128 GiB of system RAM; we run eval
  with `fast_test: 1000` to keep it tractable.

## One-time setup

### 1. Create the venv and install Python dependencies

```bash
python3.10 -m venv nbfnet-venv
nbfnet-venv/bin/pip install -r NBFNet/requirements.txt
```

This pulls PyTorch 2.9.0+cu128 (native Blackwell support), TorchDrug
0.2.1, PyG, and the rest. `numpy` is pinned `< 2` because `rdkit`
(transitively required by TorchDrug) is not numpy-2 compatible.

### 2. Install a CUDA toolkit

You need `nvcc` >= 12.8 because the JIT-compiled rspmm kernel must target
sm_120. The `nvidia-*-cu12` wheels installed by pip do **not** include
the compiler; they only ship runtime `.so`s.

Why nvcc is needed at all: TorchDrug doesn't ship its key kernel
(`generalized_rspmm`) as a precompiled binary. It only ships the source
(`rspmm.cu`, `spmm.cu`), which JIT-compiles on first use. JIT compilation
needs `nvcc`.

Either install CUDA system-wide via your distro, or extract a CUDA
runfile into a private directory. I used:

```
/home/hodgesf/cuda-12.9/
```

`run.sh` reads `CUDA_HOME` from this location and prepends
`$CUDA_HOME/bin` to `$PATH` only inside the subshell, so it does not
affect your system toolchain. Edit `CUDA_HOME` in `run.sh` to match
where you installed yours.

### 3. Apply the torchdrug patches

The pip-published torchdrug 0.2.1 predates several PyTorch 2.x API
changes. Four files inside the installed torchdrug need to be replaced
before anything will import or load checkpoints correctly:

```bash
./patches/apply_patches.sh
```

This copies the patched versions from `patches/torchdrug/` into the
corresponding paths inside the venv. The four patches are:

| File | What changed | Why |
|---|---|---|
| `torchdrug/utils/torch.py` | `cpp_extension` keyword args | torch 2.9 inserted `extra_sycl_cflags` into the signature, breaking positional callers |
| `torchdrug/layers/functional/extension/spmm.h` | `ATen/native/SparseTensorUtils.h` | header was moved out of `ATen/` in newer torch |
| `torchdrug/layers/functional/extension/rspmm.h` | same as above | same as above |
| `torchdrug/core/engine.py` | `weights_only=False` + non-tensor `state_dict` filter | torch 2.6 flipped `torch.load` default; torchdrug stores `Graph` objects as buffers which PyTorch 2.x's `load_state_dict` can't process |

You only need to run the patch script once per venv (after a fresh
`pip install`).

### 4. CUDA-library headers via `CPATH`

TorchDrug's JIT-compiled CUDA sources `#include` cuSPARSE, cuBLAS, and
cuSolver headers. A bare `nvcc` install does not ship those headers —
they live inside the `nvidia-*-cu12` pip wheels that PyTorch installed
in your venv.

`CPATH` is an environment variable the C/C++/CUDA compiler reads at
compile time, treated as a list of `-isystem` directories. When `nvcc`
compiles `rspmm.cu` and encounters `#include <cusparse.h>`, it walks
`CPATH` looking for the header.

`run.sh` exports `CPATH` pointing at the relevant venv subdirectories:

```bash
NV=$VENV/lib/python3.10/site-packages/nvidia
export CPATH=$NV/cusparse/include:$NV/cublas/include:$NV/cusolver/include:$NV/cuda_runtime/include:$NV/cuda_nvrtc/include
```

With this in place, the rspmm/spmm kernels JIT-compile successfully on
first run. Nothing to do here yourself — `run.sh` handles it.

## Data preparation

### 1. Get the raw RDKG

Place the two source files in `data/`:

```
data/all_nodes.jsonl     RDKG node records (id, category, description, ...)
data/all_edges.jsonl     RDKG edge records (subject, predicate, object, knowledge_level, ...)
```

The raw graph is ~1.59M nodes and ~13.93M edges; the edge file is ~8.7 GB.

### 2. Prune the graph

```bash
python scripts/prune_graph.py \
    --nodes-in data/all_nodes.jsonl \
    --edges-in data/all_edges.jsonl \
    --nodes-out data/pruned_nodes.jsonl \
    --edges-out data/pruned_edges.jsonl
```

This runs a six-step pipeline:

1. Drop ontology-scaffolding predicates: `subclass_of`, `related_to`,
   `equivalent_to`, `orthologous_to`, `member_of`.
2. Drop edges with `knowledge_level == "prediction"` (already-predicted
   facts we don't want to train on).
3. Drop self-loops.
4. Drop GO super-hubs (any GO node with degree > 1000).
5. BFS 6 hops from Gene/Protein and Disease anchor seeds; drop any node
   not reached.
6. Drop nodes with zero surviving edges.

Result: ~938K nodes, ~2.93M edges. Peak memory ~1 GB.

### 3. Build the splits

```bash
python scripts/build_splits.py \
    --nodes-in data/pruned_nodes.jsonl \
    --edges-in data/pruned_edges.jsonl \
    --out-dir splits/ \
    --seed 42
```

This identifies the six target predicates (`associated_with`,
`contributes_to`, `correlated_with`, `positively_correlated_with`,
`negatively_correlated_with`, `causes`), groups gene-disease target
edges by unordered pair, and performs an 80/10/10 split **at the pair
level** so every edge between a held-out pair lands in the same split.
The transductive filter then drops valid/test edges whose endpoints
don't appear in train.

Four files are written to `splits/`:

| File | Purpose | Approx size |
|---|---|---|
| `train.txt` | propagation graph (every surviving edge except held-out pair edges) | 2,860,161 edges |
| `train_targets.txt` | supervision set (target edges in train pairs only) | 275,447 edges |
| `valid.txt` | held-out target edges (valid pairs) | 33,849 edges |
| `test.txt` | held-out target edges (test pairs) | 33,695 edges |

The decoupling of `train.txt` from `train_targets.txt` is the key
methodological move: messages propagate over the full pruned graph, but
the BCE loss is computed only on gene-disease target edges. This avoids
the supervision mismatch that previously plateaued MRR at ~0.02 (when
the 90%-non-target gradient signal pushed the model to predict
chemical-protein interactions instead of gene-disease associations).

## Training

The training entry point is `run.sh`, which wraps `NBFNet/script/run.py`
with the CUDA env vars described above. The config lives at
`NBFNet/config/knowledge_graph/rdkg.yaml`.

```bash
./run.sh -c NBFNet/config/knowledge_graph/rdkg.yaml --gpus '[0]'
```

Notes on the command:

- The `'[0]'` is quoted because zsh would otherwise glob `[0]` as a
  character class.
- Checkpoints are written to
  `~/experiments/KnowledgeGraphCompletion/RDKG/NBFNet/<timestamp>/model_epoch_N.pth`.
- On RTX PRO 4000 Blackwell (24 GiB), one epoch takes ~9 hours at
  batch size 4, hidden dim 16, 6 layers. Peak GPU memory ~12 GiB.
- For long runs, launch inside `screen` or `tmux` so the process
  survives terminal disconnects.

### Resuming from a checkpoint

To continue training from an existing checkpoint, add a `checkpoint:`
field at the top of the yaml pointing at the `.pth` file, and set
`train.num_epoch` to the number of *additional* epochs you want. The
solver loads the model weights and Adam optimizer state automatically;
only the epoch counter resets to zero (so the resumed run's
`model_epoch_1.pth` is actually your cumulative-epoch-N+1).

A new timestamped experiment directory is created on every launch, so
resumed runs do not overwrite prior ones.

## Evaluation

To evaluate a saved checkpoint without further training, set
`train.num_epoch: 0` in the yaml and add `checkpoint:` pointing at the
`.pth` file. `run.py`'s `train_and_validate()` short-circuits when
`num_epoch == 0`, and the subsequent `test()` call evaluates the loaded
model on both valid and test splits.

Evaluation is **filtered ranking** against the full entity set: for
each held-out triple `(h, r, t)`, the model scores all 936,859 candidate
tails, filters out other known true triples, and ranks `t`. Reported
metrics: MRR, mean rank, Hits@1, Hits@3, Hits@10.

### Memory note

`full_batch_eval: yes` against the full 33K valid/test set currently
exceeds 128 GiB of system RAM (Linux OOM-kills the process). The yaml
includes `fast_test: 1000`, which subsamples valid/test to 1000 queries
each (deterministic, seed 1024). This makes evaluation finish in a few
minutes and keeps RAM usage trivial. The metrics in this mode are
unbiased estimates of the full-corpus filtered ranking on the same
checkpoint.

To run the genuine full-corpus eval, drop the `fast_test` line and
either close all other RAM consumers or batch the eval queries.

## Running the analyzer

`analyze_rdkg.py` is a streaming profiler that produces the summary
stats in `stats.txt` (node bucket distribution, predicate breakdown,
degree histograms, metaedge counts, gene-disease focus). It works on
both the raw and pruned graphs:

```bash
python analyze_rdkg.py --nodes data/all_nodes.jsonl    --edges data/all_edges.jsonl
python analyze_rdkg.py --nodes data/pruned_nodes.jsonl --edges data/pruned_edges.jsonl
```

It is a single streaming pass per file (the edges file is read once),
and peak memory is roughly the number of unique node IDs.
