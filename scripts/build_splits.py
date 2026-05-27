"""build_splits.py - construct decoupled propagation/supervision splits for
NBFNet training on the pruned RDKG.

Output files (all TSV with `head<TAB>predicate<TAB>tail`, no header):
  train.txt           Full propagation graph: every surviving edge EXCEPT
                      those connecting any (gene, disease) pair held out
                      for valid/test. Used as the message-passing graph
                      at every BF layer.
  train_targets.txt   Subset of train.txt: just the gene-disease target
                      edges whose pair is in the train split. Used as
                      training supervision (the loss is computed on these).
  valid.txt           Gene-disease target edges whose pair is in the
                      valid split. NOT in train.txt -- held out.
  test.txt            Same as valid but for the test split.

Target = (Gene/Protein, t_pred, Disease) or (Disease, t_pred, Gene/Protein)
where t_pred is one of:
    biolink:associated_with
    biolink:contributes_to
    biolink:correlated_with
    biolink:positively_correlated_with
    biolink:negatively_correlated_with
    biolink:causes

Splits happen at the PAIR level (unordered {gene, disease}), not edge level:
all edges between a held-out pair go to the same split, so the model
never sees a direct edge between a query's endpoints during training.
This is the leakage guard.

After pair-level split, valid/test are filtered so every entity in them
also appears in train.txt (transductive requirement).
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from time import time


TARGET_PREDICATES = frozenset({
    "biolink:associated_with",
    "biolink:contributes_to",
    "biolink:correlated_with",
    "biolink:positively_correlated_with",
    "biolink:negatively_correlated_with",
    "biolink:causes",
})


def bucket(category_list):
    """Coarse biolink category bucket. Same logic as prune_graph.py so the
    Gene/Disease anchor classification is identical."""
    if isinstance(category_list, str):
        cats = {category_list}
    else:
        cats = set(category_list or [])
    if cats & {"biolink:Gene", "biolink:Protein"}:
        return "Gene/Protein"
    if cats & {"biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"}:
        return "Disease"
    return "Other"


def log(*a, **k):
    print(*a, file=sys.stderr, flush=True, **k)


def write_tsv(path, edges):
    """Write a list of (s, p, o) triples as a TSV file."""
    with open(path, "w") as f:
        for s, p, o in edges:
            f.write(f"{s}\t{p}\t{o}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes-in", required=True,
                        help="pruned_nodes.jsonl")
    parser.add_argument("--edges-in", required=True,
                        help="pruned_edges.jsonl")
    parser.add_argument("--out-dir", required=True,
                        help="directory to write train.txt / train_targets.txt"
                             " / valid.txt / test.txt")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the split (default 42)")
    parser.add_argument("--valid-frac", type=float, default=0.1,
                        help="fraction of target pairs for valid (default 0.1)")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="fraction of target pairs for test (default 0.1)")
    args = parser.parse_args()

    # ----- Phase A: read pruned nodes -> bucket map -----
    log("Phase A: reading nodes...")
    t = time()
    node_bucket = {}
    with open(args.nodes_in) as f:
        for line in f:
            rec = json.loads(line)
            node_bucket[rec["id"]] = bucket(rec.get("category"))
    log(f"  {len(node_bucket):,} nodes ({time()-t:.1f}s)")

    # ----- Phase B: read pruned edges, classify each as target / non-target -----
    log("Phase B: reading edges, classifying targets...")
    t = time()
    # We store ALL edges as a flat list of tuples. For 2.93M edges that's
    # ~1 GB of Python objects -- fine.
    all_edges = []        # list of (s, p, o)
    target_idx = []       # indices into all_edges that are target edges
    # Group target edges by unordered pair so we can split at the pair level.
    pair_to_target_idx = defaultdict(list)
    with open(args.edges_in) as f:
        for line in f:
            rec = json.loads(line)
            s, p, o = rec["subject"], rec["predicate"], rec["object"]
            idx = len(all_edges)
            all_edges.append((s, p, o))
            if p not in TARGET_PREDICATES:
                continue
            sb = node_bucket.get(s, "Other")
            ob = node_bucket.get(o, "Other")
            is_gd = (
                (sb == "Gene/Protein" and ob == "Disease") or
                (sb == "Disease" and ob == "Gene/Protein")
            )
            if not is_gd:
                continue
            target_idx.append(idx)
            pair = tuple(sorted([s, o]))   # unordered (gene, disease) pair
            pair_to_target_idx[pair].append(idx)
    log(f"  {len(all_edges):,} edges read ({time()-t:.1f}s)")
    log(f"  {len(target_idx):,} target edges (gene<->disease with target predicate)")
    log(f"  {len(pair_to_target_idx):,} unique gene-disease pairs")

    # ----- Phase C: split pairs into train / valid / test -----
    log("Phase C: pair-level split...")
    rng = random.Random(args.seed)
    pairs = sorted(pair_to_target_idx.keys())  # deterministic ordering
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * args.test_frac))
    n_valid = int(round(n * args.valid_frac))
    n_train = n - n_test - n_valid
    train_pairs = set(pairs[:n_train])
    valid_pairs = set(pairs[n_train:n_train + n_valid])
    test_pairs = set(pairs[n_train + n_valid:])
    log(f"  train: {len(train_pairs):,} pairs")
    log(f"  valid: {len(valid_pairs):,} pairs")
    log(f"  test:  {len(test_pairs):,} pairs")

    # ----- Phase D: assign edges to splits -----
    log("Phase D: assigning edges to splits...")
    # Build a fast lookup from edge index -> which split it's in (target only).
    # Non-target edges always go to train.
    edge_split = {}  # idx -> "valid" or "test" or "train"
    for idx in target_idx:
        s, _, o = all_edges[idx]
        pair = tuple(sorted([s, o]))
        if pair in valid_pairs:
            edge_split[idx] = "valid"
        elif pair in test_pairs:
            edge_split[idx] = "test"
        else:
            edge_split[idx] = "train"

    train_edges = []      # all non-target edges + target edges in train pairs
    train_targets = []    # target edges in train pairs (subset of train_edges)
    valid_edges = []      # target edges in valid pairs
    test_edges = []       # target edges in test pairs

    for idx, e in enumerate(all_edges):
        split = edge_split.get(idx)
        if split == "valid":
            valid_edges.append(e)
        elif split == "test":
            test_edges.append(e)
        elif split == "train":
            # target edge in train -- include in both train.txt and train_targets
            train_edges.append(e)
            train_targets.append(e)
        else:
            # non-target edge -- always goes to propagation graph (train.txt)
            train_edges.append(e)

    log(f"  train.txt: {len(train_edges):,} edges (propagation graph)")
    log(f"  train_targets.txt: {len(train_targets):,} edges (training supervision)")
    log(f"  valid.txt: {len(valid_edges):,} edges (before transductive filter)")
    log(f"  test.txt:  {len(test_edges):,} edges (before transductive filter)")

    # ----- Phase E: enforce transductive setup ------
    # Every entity in valid/test must also appear in train.txt; otherwise the
    # model has no way to embed it at query time. Drop valid/test edges that
    # touch any such "orphan" entity.
    log("Phase E: transductive filter (every valid/test entity must be in train)...")
    train_entities = set()
    for s, _, o in train_edges:
        train_entities.add(s)
        train_entities.add(o)

    def filter_transductive(edges, name):
        keep = []
        dropped_edges = 0
        for s, p, o in edges:
            if s in train_entities and o in train_entities:
                keep.append((s, p, o))
            else:
                dropped_edges += 1
        log(f"  {name}: {len(keep):,} kept, {dropped_edges:,} dropped "
            f"(endpoint not in train.txt)")
        return keep

    valid_edges = filter_transductive(valid_edges, "valid")
    test_edges = filter_transductive(test_edges, "test")

    # ----- Phase F: leakage check -----
    # No (h, t) pair in valid/test should have any edge in train.txt.
    log("Phase F: leakage check...")
    train_pair_set = set()
    for s, _, o in train_edges:
        train_pair_set.add(tuple(sorted([s, o])))
    leaked_v = sum(1 for s, _, o in valid_edges
                   if tuple(sorted([s, o])) in train_pair_set)
    leaked_t = sum(1 for s, _, o in test_edges
                   if tuple(sorted([s, o])) in train_pair_set)
    log(f"  valid edges sharing a pair with train: {leaked_v:,}")
    log(f"  test  edges sharing a pair with train: {leaked_t:,}")
    if leaked_v > 0 or leaked_t > 0:
        log("  WARNING: leakage detected. Pair-level split should make this 0; "
            "investigate.")

    # ----- Phase G: write outputs -----
    log("Phase G: writing outputs...")
    os.makedirs(args.out_dir, exist_ok=True)
    write_tsv(os.path.join(args.out_dir, "train.txt"), train_edges)
    write_tsv(os.path.join(args.out_dir, "train_targets.txt"), train_targets)
    write_tsv(os.path.join(args.out_dir, "valid.txt"), valid_edges)
    write_tsv(os.path.join(args.out_dir, "test.txt"), test_edges)
    log(f"  wrote {args.out_dir}/train.txt         ({len(train_edges):,} lines)")
    log(f"  wrote {args.out_dir}/train_targets.txt ({len(train_targets):,} lines)")
    log(f"  wrote {args.out_dir}/valid.txt         ({len(valid_edges):,} lines)")
    log(f"  wrote {args.out_dir}/test.txt          ({len(test_edges):,} lines)")
    log("DONE")


if __name__ == "__main__":
    main()
