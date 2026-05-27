"""prune_graph.py - prune RDKG via the agreed pipeline.

Pipeline (in order):
  1. Drop edges with predicate in {subclass_of, related_to, equivalent_to,
     orthologous_to, member_of}.
  2. Drop edges with knowledge_level == "prediction".
  3. Drop self-loop edges (subject == object).
  4. Drop GO nodes (and their edges) whose surviving degree exceeds the
     `--go-hub-degree` threshold (default 1000).
  5. BFS `--hops` hops (default 6) from Gene/Protein + Disease anchors;
     drop any node that BFS does not reach. Drop edges with either endpoint
     dropped.
  6. Drop nodes that ended up with zero surviving edges (final isolated sweep).

Inputs:  all_nodes.jsonl, all_edges.jsonl  (JSONL, one record per line).
Outputs: pruned_nodes.jsonl, pruned_edges.jsonl  (same schemas, surviving
         records are written verbatim from the input lines).

Implementation: streams the edge file twice (the big one), the node file
twice (the small one). The only RAM-resident structures are integer adjacency
for BFS and a few node-id <-> int dicts -- peak memory ~1 GB on the full
RDKG.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from time import time


# Predicates to drop. Ontology scaffolding and other low-signal predicates per
# the decisions in our planning discussion.
DROP_PREDICATES = frozenset({
    "biolink:subclass_of",
    "biolink:related_to",
    "biolink:equivalent_to",
    "biolink:orthologous_to",
    "biolink:member_of",
})

# Knowledge levels to drop -- "prediction" edges are model-inferred, we want
# only asserted facts in the training graph.
DROP_KNOWLEDGE_LEVELS = frozenset({"prediction"})


def bucket(category_list):
    """Coarse biolink category bucket. Same logic as analyze_rdkg.py so the
    Gene/Disease anchor sets are identical to what we discussed."""
    if isinstance(category_list, str):
        cats = {category_list}
    else:
        cats = set(category_list or [])
    if cats & {"biolink:Gene", "biolink:Protein"}:
        return "Gene/Protein"
    if cats & {"biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"}:
        return "Disease"
    if "biolink:PhenotypicFeature" in cats:
        return "Phenotype"
    if cats & {"biolink:ChemicalEntity", "biolink:Drug", "biolink:SmallMolecule",
               "biolink:MolecularEntity"}:
        return "Chemical/Drug"
    if cats & {"biolink:BiologicalProcess", "biolink:MolecularFunction",
               "biolink:CellularComponent", "biolink:GeneOntologyClass"}:
        return "GO"
    if "biolink:Pathway" in cats:
        return "Pathway"
    if cats & {"biolink:AnatomicalEntity", "biolink:GrossAnatomicalStructure",
               "biolink:Cell"}:
        return "Anatomy"
    if "biolink:Publication" in cats:
        return "Publication"
    return "Other"


def passes_basic_filters(rec):
    """Returns True if the edge survives steps 1-3 (predicate, knowledge
    level, self-loop). Called twice -- once during the in-memory pass and once
    when writing the output -- so it lives in a function."""
    if rec.get("predicate") in DROP_PREDICATES:
        return False
    if rec.get("knowledge_level") in DROP_KNOWLEDGE_LEVELS:
        return False
    if rec.get("subject") == rec.get("object"):
        return False
    return True


def log(*a, **k):
    """All progress messages go to stderr so stdout stays clean for redirect."""
    print(*a, file=sys.stderr, flush=True, **k)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes-in", required=True, help="all_nodes.jsonl")
    parser.add_argument("--edges-in", required=True, help="all_edges.jsonl")
    parser.add_argument("--nodes-out", required=True, help="pruned_nodes.jsonl")
    parser.add_argument("--edges-out", required=True, help="pruned_edges.jsonl")
    parser.add_argument("--hops", type=int, default=6,
                        help="BFS hop budget from anchors (default 6)")
    parser.add_argument("--go-hub-degree", type=int, default=1000,
                        help="drop GO nodes whose degree exceeds this after "
                             "step 1-3 (default 1000)")
    args = parser.parse_args()

    # =============================================================
    # PHASE A: stream nodes once, build vocab and anchor/GO flags
    # =============================================================
    log("Phase A: streaming nodes...")
    t = time()
    # node_id_to_int is the CURIE -> integer ID map. We carry integer IDs
    # through the in-memory phases because the dict + adjacency are large.
    node_id_to_int = {}
    is_anchor = {}    # int -> bool  (Gene/Protein or Disease)
    is_go = {}        # int -> bool  (CURIE has "GO:" prefix)
    with open(args.nodes_in) as f:
        for line in f:
            rec = json.loads(line)
            nid = rec["id"]
            int_id = len(node_id_to_int)
            node_id_to_int[nid] = int_id
            b = bucket(rec.get("category"))
            is_anchor[int_id] = (b == "Gene/Protein" or b == "Disease")
            is_go[int_id] = nid.startswith("GO:")
    n_anchors = sum(1 for v in is_anchor.values() if v)
    log(f"  total nodes: {len(node_id_to_int):,}  ({time()-t:.1f}s)")
    log(f"  Gene/Protein + Disease anchors: {n_anchors:,}")

    # =============================================================
    # PHASE B: stream edges once, apply basic filters, collect (s,o) pairs
    # =============================================================
    log("Phase B: streaming edges, applying basic filters (predicate / "
        "knowledge_level / self-loop) ...")
    t = time()
    # Two parallel arrays of int IDs. Cheaper than a list of tuples.
    subj_arr = []
    obj_arr = []
    counts = Counter()
    with open(args.edges_in) as f:
        for line in f:
            rec = json.loads(line)
            counts["total"] += 1
            if not passes_basic_filters(rec):
                counts["dropped_basic"] += 1
                continue
            s = rec["subject"]
            o = rec["object"]
            s_int = node_id_to_int.get(s)
            o_int = node_id_to_int.get(o)
            # If the edge references a CURIE that isn't in the nodes file,
            # we have no category info and can't BFS through it -- drop.
            if s_int is None or o_int is None:
                counts["missing_node_vocab"] += 1
                continue
            subj_arr.append(s_int)
            obj_arr.append(o_int)
            if counts["total"] % 1_000_000 == 0:
                log(f"  edges read: {counts['total']:,}  ({time()-t:.1f}s)")
    log(f"  input edges: {counts['total']:,}")
    log(f"  dropped (predicate/k-level/self-loop): {counts['dropped_basic']:,}")
    log(f"  dropped (subject or object not in node vocab): "
        f"{counts['missing_node_vocab']:,}")
    log(f"  surviving after steps 1-3: {len(subj_arr):,}")

    # =============================================================
    # PHASE C: count degrees, identify GO super-hubs
    # =============================================================
    log("Phase C: counting degrees, identifying GO super-hubs...")
    deg = Counter()
    for s_int, o_int in zip(subj_arr, obj_arr):
        deg[s_int] += 1
        deg[o_int] += 1
    superhubs = set()
    for int_id, d in deg.items():
        if is_go.get(int_id, False) and d > args.go_hub_degree:
            superhubs.add(int_id)
    log(f"  GO super-hubs (degree > {args.go_hub_degree}): {len(superhubs):,}")
    if superhubs:
        # Print the worst offenders so we can see the cut is sensible.
        int_to_id = {v: k for k, v in node_id_to_int.items()}
        ranked = sorted(((deg[i], i) for i in superhubs), reverse=True)[:20]
        log("  top GO super-hubs being dropped (degree, CURIE):")
        for d, int_id in ranked:
            log(f"    {d:>10,d}  {int_to_id[int_id]}")
    del deg  # free

    # =============================================================
    # PHASE D: build adjacency, excluding super-hub endpoints
    # =============================================================
    log("Phase D: building adjacency (excluding super-hub edges)...")
    t = time()
    adj = defaultdict(list)
    n_after_d = 0
    n_dropped_superhub = 0
    for s_int, o_int in zip(subj_arr, obj_arr):
        if s_int in superhubs or o_int in superhubs:
            n_dropped_superhub += 1
            continue
        # Undirected adjacency: BFS traverses both directions.
        adj[s_int].append(o_int)
        adj[o_int].append(s_int)
        n_after_d += 1
    log(f"  dropped (super-hub endpoint): {n_dropped_superhub:,}")
    log(f"  surviving after step 4: {n_after_d:,}")
    log(f"  ({time()-t:.1f}s)")
    # Free the flat arrays -- adjacency has all we need for BFS.
    del subj_arr
    del obj_arr

    # =============================================================
    # PHASE E: BFS from anchor seeds (Gene/Protein + Disease nodes)
    # =============================================================
    seeds = {i for i, anchor in is_anchor.items()
             if anchor and i not in superhubs}
    log(f"Phase E: BFS {args.hops} hops from {len(seeds):,} anchor seeds...")
    reached = set(seeds)
    frontier = seeds
    for hop in range(1, args.hops + 1):
        new_frontier = set()
        for node in frontier:
            for nb in adj[node]:
                if nb not in reached:
                    new_frontier.add(nb)
        reached |= new_frontier
        log(f"  hop {hop}: +{len(new_frontier):,} new, "
            f"{len(reached):,} total reached")
        frontier = new_frontier
        if not frontier:
            break
    log(f"  total reached after BFS: {len(reached):,}")
    del adj  # free, no longer needed

    # =============================================================
    # PHASE F: stream edges again, write surviving ones to disk
    # =============================================================
    # An edge survives iff:
    #   - passes basic filters (predicate / k-level / self-loop)
    #   - both endpoints are in the node vocab
    #   - neither endpoint is a super-hub
    #   - both endpoints are in `reached`
    # We re-check all filters here rather than carry state from phase B,
    # so we're sure pruned_edges.jsonl is an exact filtered subset of input.
    log("Phase F: writing pruned edges (second pass over edges file)...")
    t = time()
    final_node_ints = set()  # nodes that appear in at least one surviving edge
    n_in = 0
    n_out = 0
    with open(args.edges_in) as fin, open(args.edges_out, "w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            if not passes_basic_filters(rec):
                continue
            s_int = node_id_to_int.get(rec["subject"])
            o_int = node_id_to_int.get(rec["object"])
            if s_int is None or o_int is None:
                continue
            if s_int in superhubs or o_int in superhubs:
                continue
            if s_int not in reached or o_int not in reached:
                continue
            fout.write(line)
            final_node_ints.add(s_int)
            final_node_ints.add(o_int)
            n_out += 1
            if n_in % 1_000_000 == 0:
                log(f"  edges read: {n_in:,}  written: {n_out:,}  "
                    f"({time()-t:.1f}s)")
    log(f"  pruned edges written: {n_out:,}  -->  {args.edges_out}")
    log(f"  nodes with >=1 surviving edge: {len(final_node_ints):,}")

    # =============================================================
    # PHASE H: stream nodes again, write surviving ones to disk
    # =============================================================
    # We drop nodes that are not in any surviving edge -- this implicitly
    # handles step 6 (final isolated-node sweep) and any anchor that ended
    # up with zero surviving edges after the cuts above.
    log("Phase H: writing pruned nodes (second pass over nodes file)...")
    t = time()
    n_in = 0
    n_out = 0
    with open(args.nodes_in) as fin, open(args.nodes_out, "w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            int_id = node_id_to_int.get(rec["id"])
            if int_id is None:
                continue
            if int_id not in final_node_ints:
                continue
            fout.write(line)
            n_out += 1
    log(f"  pruned nodes written: {n_out:,}  -->  {args.nodes_out}")
    log(f"  ({time()-t:.1f}s)")
    log("DONE")


if __name__ == "__main__":
    main()
