"""Per-step profiler: isolates forward time, backward time, and peak GPU
memory for a single training step, with num_workers=0 so the timing is pure
compute (no dataloader overlap to confuse the picture).

Usage:
    python script/profile_step.py <config.yaml> [batch_size]

Example:
    python script/profile_step.py config/knowledge_graph/rdkg_fast.yaml 16
"""
import sys
import os
import time

import torch
import jinja2
import yaml
import easydict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchdrug import core, utils
from torch.utils import data as torch_data
from nbfnet import dataset, layer, model, task, util  # noqa: F401  (registers classes)

cfg_file = sys.argv[1]
raw = open(cfg_file).read()
cfg = easydict.EasyDict(yaml.safe_load(jinja2.Template(raw).render({"gpus": [0]})))

# optional CLI override of batch size so we can sweep without editing the yaml
if len(sys.argv) > 2:
    cfg.engine.batch_size = int(sys.argv[2])
bs = cfg.engine.batch_size

ds = core.Configurable.load_config_dict(cfg.dataset)
solver = util.build_solver(cfg, ds)

device = solver.device
mdl = solver.model
mdl.split = "train"
mdl.train()
opt = solver.optimizer

loader = torch_data.DataLoader(solver.train_set, bs, shuffle=True, num_workers=0)
it = iter(loader)

print("\n=== profiling  batch_size=%d  device=%s ===" % (bs, device))
print("(step 0 includes one-time setup / cudnn autotune; read steps 1+)")
for step in range(6):
    batch = utils.cuda(next(it), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(); t0 = time.time()
    loss, metric = mdl(batch)
    torch.cuda.synchronize(); t1 = time.time()
    loss.backward()
    torch.cuda.synchronize(); t2 = time.time()
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); t3 = time.time()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print("step %d:  fwd %6.3fs   bwd %6.3fs   opt %6.3fs   | total %6.3fs   peak %5.1f GB"
          % (step, t1 - t0, t2 - t1, t3 - t2, t3 - t0, peak))
