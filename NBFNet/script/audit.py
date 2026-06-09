"""Faithfulness audits on a trained MaskedNBFNet checkpoint.

Three audits plus a baseline (control).

  baseline: the unmodified trained model. Reproduces the standard MRR
            this checkpoint achieves. Use this as the control.

  shuffle:  Subset-identity-shuffle. Computes all masks in the batch,
            then swaps them across queries (mask of query B applied to
            query A). If the masked subgraph is genuinely query-specific,
            MRR should crater. If MRR is preserved, the model is using
            generic structure rather than per-query selection.

  random:   Replace the learned mask with a random binary mask of the
            same total density (same k_used). If MRR survives, density
            alone matters, not the specific choices.

  permute:  Take the learned mask, then relocate a fixed fraction of
            selected edges to random other edges in the graph
            (preserving density). If MRR survives, the specific edges
            do not matter, only their density.

Usage:
    python NBFNet/script/audit.py \\
        -c NBFNet/config/knowledge_graph/rdkg_masked.yaml \\
        --checkpoint /path/to/model_epoch_N.pth \\
        --audit shuffle \\
        --gpus '[0]'
"""

import os
import sys
import argparse
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model as nbfnet_model, task, util


@nbfnet_model.R.register("model.ShuffledMaskedNBFNet")
class ShuffledMaskedNBFNet(nbfnet_model.MaskedNBFNet):
    """Subset-identity-shuffle with a cross-batch mask bank.

    With full_batch_eval=True, eval feeds one query per forward call (B=1),
    so within-batch shuffle is a no-op. Instead we maintain a rolling bank
    of recently seen query masks. Each query is scored with a swap_mask
    drawn from the bank (excluding any entry that matches its own query),
    then its own mask is pushed into the bank. After the bank fills
    (BANK_SIZE queries), every subsequent forward call applies a mask from
    a genuinely different query. The first query has an empty bank and
    falls back to its own mask -- a single-query contamination negligible
    over 1000-query eval.
    """

    BANK_SIZE = 64

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if not hasattr(self, "_bank"):
            self._bank = []  # list of (h_int, r_int, mask_tensor)

        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        assert graph.num_relation, "ShuffledMaskedNBFNet requires a relational graph"
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        B = h_index.shape[0]
        scores = []
        for b in range(B):
            h_b_int = h_index[b, 0].item()
            r_b_int = r_index[b, 0].item()

            logits = self.score_edges(graph, h_index[b, 0], r_index[b, 0])
            own_mask = self.select_mask(logits)

            candidates = [m for (h, r, m) in self._bank if (h, r) != (h_b_int, r_b_int)]
            if candidates:
                swap_mask = candidates[torch.randint(0, len(candidates), (1,)).item()]
            else:
                swap_mask = own_mask

            if len(self._bank) < self.BANK_SIZE:
                self._bank.append((h_b_int, r_b_int, own_mask.detach().clone()))
            else:
                idx = torch.randint(0, self.BANK_SIZE, (1,)).item()
                self._bank[idx] = (h_b_int, r_b_int, own_mask.detach().clone())

            g_b = self.apply_mask(graph, swap_mask)
            out = self.bellmanford(g_b, h_index[b, 0].unsqueeze(0), r_index[b, 0].unsqueeze(0))
            feat = out["node_feature"].squeeze(1)
            scores.append(self.mlp(feat[t_index[b]]).squeeze(-1))

        return torch.stack(scores, dim=0).view(shape)


@nbfnet_model.R.register("model.RandomMaskedNBFNet")
class RandomMaskedNBFNet(nbfnet_model.MaskedNBFNet):
    """Random mask of same density as the learned mask."""

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        assert graph.num_relation, "RandomMaskedNBFNet requires a relational graph"
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        B = h_index.shape[0]
        scores = []
        for b in range(B):
            logits = self.score_edges(graph, h_index[b, 0], r_index[b, 0])
            learned = self.select_mask(logits)
            k = int(learned.sum().item())
            E = learned.numel()
            device = learned.device

            rand = torch.zeros(E, device=device)
            if k > 0:
                idx = torch.randperm(E, device=device)[:k]
                rand[idx] = 1.0

            g_b = self.apply_mask(graph, rand)
            out = self.bellmanford(g_b, h_index[b, 0].unsqueeze(0), r_index[b, 0].unsqueeze(0))
            feat = out["node_feature"].squeeze(1)
            scores.append(self.mlp(feat[t_index[b]]).squeeze(-1))

        return torch.stack(scores, dim=0).view(shape)


@nbfnet_model.R.register("model.PermutedMaskedNBFNet")
class PermutedMaskedNBFNet(nbfnet_model.MaskedNBFNet):
    """Permute a fraction of selected edges to random unselected edges.

    Configurable via permute_fraction in the yaml task.model block.
    Default 0.5 (swap half the learned mask).
    """

    def __init__(self, *args, permute_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.permute_fraction = float(permute_fraction)

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        assert graph.num_relation, "PermutedMaskedNBFNet requires a relational graph"
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        B = h_index.shape[0]
        scores = []
        for b in range(B):
            logits = self.score_edges(graph, h_index[b, 0], r_index[b, 0])
            learned = self.select_mask(logits)

            sel = learned.nonzero().squeeze(-1)
            unsel = (1 - learned).nonzero().squeeze(-1)
            device = learned.device

            n_swap = int(sel.numel() * self.permute_fraction)
            mask = learned.clone()
            if n_swap > 0 and unsel.numel() >= n_swap:
                drop = sel[torch.randperm(sel.numel(), device=device)[:n_swap]]
                add = unsel[torch.randperm(unsel.numel(), device=device)[:n_swap]]
                mask[drop] = 0
                mask[add] = 1

            g_b = self.apply_mask(graph, mask)
            out = self.bellmanford(g_b, h_index[b, 0].unsqueeze(0), r_index[b, 0].unsqueeze(0))
            feat = out["node_feature"].squeeze(1)
            scores.append(self.mlp(feat[t_index[b]]).squeeze(-1))

        return torch.stack(scores, dim=0).view(shape)


AUDIT_TO_CLASS = {
    "baseline": "MaskedNBFNet",
    "shuffle":  "ShuffledMaskedNBFNet",
    "random":   "RandomMaskedNBFNet",
    "permute":  "PermutedMaskedNBFNet",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="yaml config used to train the checkpoint")
    parser.add_argument("--checkpoint", required=True,
                        help="path to trained .pth")
    parser.add_argument("--audit", required=True,
                        choices=list(AUDIT_TO_CLASS.keys()))
    parser.add_argument("--gpus", default="[0]")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    cfg = util.load_config(args.config, context={"gpus": eval(args.gpus)})

    # Swap the model class to the audit variant
    cfg.task.model["class"] = AUDIT_TO_CLASS[args.audit]

    # If the yaml has a checkpoint: line for resume-training, drop it.
    # We load the audit checkpoint manually below.
    if "checkpoint" in cfg:
        del cfg["checkpoint"]

    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("=" * 70)
        logger.warning("FAITHFULNESS AUDIT: %s" % args.audit.upper())
        logger.warning("Model class: %s" % cfg.task.model["class"])
        logger.warning("Config: %s" % args.config)
        logger.warning("Checkpoint: %s" % args.checkpoint)
        logger.warning("Seed: %d" % args.seed)
        logger.warning("=" * 70)

    dataset_obj = core.Configurable.load_config_dict(cfg.dataset)
    # util.build_solver only injects num_entities when class == "MaskedNBFNet".
    # Our audit subclasses (ShuffledMaskedNBFNet etc.) skip that path, so the
    # EdgeSelector's nn.Embedding fails on num_entities=None. Inject manually.
    cfg.task.model["num_entities"] = dataset_obj.num_entity
    solver = util.build_solver(cfg, dataset_obj)
    solver.load(args.checkpoint)

    solver.model.split = "valid"
    valid_metric = solver.evaluate("valid")

    solver.model.split = "test"
    test_metric = solver.evaluate("test")

    if comm.get_rank() == 0:
        logger.warning("=" * 70)
        logger.warning("AUDIT %s -- FINAL RESULTS" % args.audit.upper())
        logger.warning("VALID: %s" % pprint.pformat(dict(valid_metric)))
        logger.warning("TEST:  %s" % pprint.pformat(dict(test_metric)))
        logger.warning("=" * 70)


if __name__ == "__main__":
    main()
