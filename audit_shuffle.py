"""Subset-identity-shuffle audit (decision 10).

The faithfulness test: if MaskedNBFNet has learned the right causal edges, then
permuting entity IDs *within* the selected subgraph should collapse the
prediction toward chance. If MRR survives the shuffle, the predictor isn't
actually using the selected edges' identities -- the cold-start junk equilibrium.

Usage:
    python audit_shuffle.py <config.yaml> <checkpoint.pth> [--num-queries N]

Requires:
  - A trained MaskedNBFNet checkpoint.
  - The same yaml used for training (so the model + data load identically).

Output for each query:
  - Original predicted score for the true tail.
  - Shuffled-subgraph predicted score for the true tail.
  - Rank under original vs under shuffle.
  - Aggregate: MRR over the sampled queries, original vs shuffled.

If shuffled MRR collapses (e.g., toward 1/|V|): faithfulness holds.
If shuffled MRR ~= original MRR: leakage -- the predictor isn't actually using
the selected subgraph causally; it's getting the answer some other way.
"""
import argparse
import os
import sys
import random

import torch
import jinja2
import yaml
import easydict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NBFNet"))

from torchdrug import core, utils
from torch.utils import data as torch_data
from nbfnet import dataset, layer, model, task, util  # registers classes


def load_solver_with_checkpoint(cfg_file, ckpt_path, gpus):
    """Mirror script/run.py setup, plus checkpoint load."""
    raw = open(cfg_file).read()
    cfg = easydict.EasyDict(yaml.safe_load(jinja2.Template(raw).render({"gpus": gpus})))
    cfg.checkpoint = ckpt_path  # ensure checkpoint loads in build_solver
    ds = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, ds)
    return solver, ds


def score_query(model, fact_graph, h, t_candidates, r):
    """Run one query through MaskedNBFNet, return scores over candidates.

    h, r: scalars. t_candidates: (num_cand,) tensor of entity ids.
    Returns: (num_cand,) tensor of scores.
    """
    # Wrap into (B=1, 1+num_neg) shape the model's forward expects.
    h_idx = h.view(1, 1).expand(1, t_candidates.numel())
    t_idx = t_candidates.view(1, -1)
    r_idx = r.view(1, 1).expand(1, t_candidates.numel())
    # all_loss=None -> eval mode (skip remove_easy_edges).
    score = model(fact_graph, h_idx, t_idx, r_idx, all_loss=None)
    return score.view(-1)


def shuffle_within_subgraph(fact_graph, selected_edge_ids_in_aug, h_id, t_id, seed):
    """Permute entity IDs WITHIN the originally-selected subgraph.

    Args:
        fact_graph: the UN-AUGMENTED graph the model was given (it re-augments
            internally inside forward).
        selected_edge_ids_in_aug: indices into the AUGMENTED edge list (this is
            what audit['selected_edge_ids'] holds, since the model's selector
            runs on the augmented graph).
        h_id, t_id: query head + true tail. NOT shuffled, otherwise we'd be
            testing a different question.
        seed: int.

    Returns:
        A new UN-AUGMENTED Graph with entity IDs permuted within the subgraph.
        The model's forward will re-augment it internally.
    """
    g = torch.Generator().manual_seed(seed)

    # Identify the node set of the originally-selected subgraph by re-augmenting
    # and indexing with the augmented-edge IDs the audit gave us.
    selected_edge_ids_in_aug = selected_edge_ids_in_aug.to(fact_graph.edge_list.device)
    aug = fact_graph.undirected(add_inverse=True)
    sub_edges_aug = aug.edge_list[selected_edge_ids_in_aug]
    sub_nodes = torch.unique(sub_edges_aug[:, :2].flatten())

    # Don't shuffle the query endpoints -- shuffling h would change the query.
    fixed_nodes = torch.tensor([int(h_id), int(t_id)], dtype=torch.long,
                               device=sub_nodes.device)
    shufflable = sub_nodes[~torch.isin(sub_nodes, fixed_nodes)]
    if shufflable.numel() < 2:
        # Nothing meaningful to shuffle (subgraph too small or only contains h/t).
        return fact_graph

    perm = shufflable[torch.randperm(shufflable.numel(), generator=g).to(shufflable.device)]

    # Full-graph permutation map: identity outside `shufflable`, random within.
    node_map = torch.arange(fact_graph.num_node, dtype=torch.long,
                            device=fact_graph.edge_list.device)
    node_map[shufflable] = perm

    # Apply to the un-augmented edge list (vectorized; correct under simultaneous
    # swaps because we don't do them iteratively).
    new_edge_list = fact_graph.edge_list.clone()
    new_edge_list[:, 0] = node_map[new_edge_list[:, 0]]
    new_edge_list[:, 1] = node_map[new_edge_list[:, 1]]

    return type(fact_graph)(
        new_edge_list,
        edge_weight=fact_graph.edge_weight,
        num_node=fact_graph.num_node,
        num_relation=fact_graph.num_relation,
        meta_dict=fact_graph.meta_dict,
        **fact_graph.data_dict,
    )


def reciprocal_rank(scores, true_idx):
    """Reciprocal rank of true_idx in scores (higher score = better)."""
    sorted_idx = torch.argsort(scores, descending=True)
    rank = (sorted_idx == true_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="yaml that trained the checkpoint")
    ap.add_argument("checkpoint", help="model_epoch_N.pth produced by training")
    ap.add_argument("--num-queries", type=int, default=20, help="how many test queries to audit")
    ap.add_argument("--gpus", nargs="*", type=int, default=[0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    solver, ds = load_solver_with_checkpoint(args.config, args.checkpoint, args.gpus)
    model_ = solver.model.model      # task.model = the MaskedNBFNet
    fact_graph = solver.model.fact_graph
    # eval mode -> deterministic top-k, no Gumbel noise
    solver.model.eval()
    assert hasattr(model_, "get_audit"), "checkpoint did not load a MaskedNBFNet"
    assert model_.audit, "model.audit must be True so we can read selected_edge_ids"

    # model's device -- query tensors must match.
    device = next(model_.parameters()).device

    # sample queries from the held-out test set
    test_set = solver.test_set
    sample_idx = random.sample(range(len(test_set)), min(args.num_queries, len(test_set)))

    orig_rrs = []
    shuf_rrs = []
    for i, idx in enumerate(sample_idx):
        h, t, r = test_set[idx].tolist()
        h_t = torch.tensor(h, device=device)
        t_t = torch.tensor(t, device=device)
        r_t = torch.tensor(r, device=device)
        candidates = torch.arange(fact_graph.num_node, device=device)  # rank against full entity set

        # 1) original: score on the real fact_graph
        with torch.no_grad():
            scores_orig = score_query(model_, fact_graph, h_t, candidates, r_t)
        # read the selected subgraph for this query from the audit
        audit = model_.get_audit()[0]
        selected = audit["selected_edge_ids"]

        # 2) shuffle identities within that subgraph and re-score
        shuffled_graph = shuffle_within_subgraph(fact_graph, selected, h, t, seed=args.seed + i)
        with torch.no_grad():
            scores_shuf = score_query(model_, shuffled_graph, h_t, candidates, r_t)

        rr_orig = reciprocal_rank(scores_orig, t)
        rr_shuf = reciprocal_rank(scores_shuf, t)
        orig_rrs.append(rr_orig)
        shuf_rrs.append(rr_shuf)
        print(f"q{i:02d}  h={h:>6}  r={r:>3}  t={t:>6}  "
              f"|sub|={selected.numel():>6}  rr_orig={rr_orig:.4f}  rr_shuf={rr_shuf:.4f}")

    mrr_orig = sum(orig_rrs) / len(orig_rrs)
    mrr_shuf = sum(shuf_rrs) / len(shuf_rrs)
    print()
    print(f"original MRR (over {len(orig_rrs)} queries): {mrr_orig:.4f}")
    print(f"shuffled MRR (over {len(shuf_rrs)} queries): {mrr_shuf:.4f}")
    print(f"chance MRR (1/|V|)                         : {1.0/fact_graph.num_node:.6f}")

    drop_ratio = mrr_shuf / max(mrr_orig, 1e-9)
    print(f"shuffled/original ratio: {drop_ratio:.3f}")
    if drop_ratio < 0.2:
        print("VERDICT: faithfulness holds -- prediction collapses under identity-shuffle.")
    elif drop_ratio < 0.6:
        print("VERDICT: partial -- some faithfulness but identity info still leaking through.")
    else:
        print("VERDICT: LEAKAGE -- prediction survives identity-shuffle; selector likely junk-equilibrium.")


if __name__ == "__main__":
    main()
