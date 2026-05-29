"""filter_nonhuman_genes.py - drop non-human gene/protein nodes from the
pruned RDKG, then drop any node left isolated by that removal.

A node is removed iff:
  - its category bucket is Gene/Protein, AND
  - it carries a taxon (in_taxon_label or taxon), AND
  - that taxon is NOT human (NCBITaxon:9606 / "Homo sapiens").

Gene/protein nodes with NO taxon field are KEPT: they are load-bearing for
the eval targets (they appear in valid/test target edges) and are most
likely human records with a missing field.

Edges with either endpoint removed are dropped. After the edge pass, any
node with zero surviving edges is dropped too (isolated sweep), matching
the final step of prune_graph.py.

Inputs:  pruned_nodes.jsonl, pruned_edges.jsonl  (JSONL, one record/line).
Outputs: same schemas, surviving records written verbatim from input lines.

Streams the edge file once and the node file twice. Only RAM-resident
structures are the non-human-gene id set and the surviving-node id set.
"""

import argparse
import json
import sys
from collections import Counter
from time import time


GENEISH = frozenset({"biolink:Gene", "biolink:Protein", "biolink:GeneOrGeneProduct"})
HUMAN = frozenset({"NCBITaxon:9606", "Homo sapiens"})


def is_geneish(category_list):
    if isinstance(category_list, str):
        cats = {category_list}
    else:
        cats = set(category_list or [])
    return bool(cats & GENEISH)


def log(*a, **k):
    print(*a, file=sys.stderr, flush=True, **k)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes-in", required=True, help="pruned_nodes.jsonl")
    parser.add_argument("--edges-in", required=True, help="pruned_edges.jsonl")
    parser.add_argument("--nodes-out", required=True,
                        help="human_nodes.jsonl (output)")
    parser.add_argument("--edges-out", required=True,
                        help="human_edges.jsonl (output)")
    args = parser.parse_args()

    # =============================================================
    # PHASE A: stream nodes once, collect non-human gene/protein ids
    # =============================================================
    log("Phase A: streaming nodes, flagging non-human gene/protein nodes...")
    t = time()
    nonhuman = set()
    counts = Counter()
    with open(args.nodes_in) as f:
        for line in f:
            rec = json.loads(line)
            counts["nodes_total"] += 1
            if not is_geneish(rec.get("category")):
                continue
            counts["gene_total"] += 1
            tax = rec.get("in_taxon_label") or rec.get("taxon")
            if tax is None:
                counts["gene_no_taxon_kept"] += 1
                continue
            if tax in HUMAN:
                counts["gene_human_kept"] += 1
                continue
            nonhuman.add(rec["id"])
    log(f"  total nodes: {counts['nodes_total']:,}")
    log(f"  gene/protein nodes: {counts['gene_total']:,}")
    log(f"    human (kept):     {counts['gene_human_kept']:,}")
    log(f"    no taxon (kept):  {counts['gene_no_taxon_kept']:,}")
    log(f"    non-human (drop): {len(nonhuman):,}")
    log(f"  ({time()-t:.1f}s)")

    # =============================================================
    # PHASE B: stream edges once, write those not touching a non-human gene
    # =============================================================
    log("Phase B: writing edges (drop any touching a non-human gene)...")
    t = time()
    surviving_nodes = set()
    n_in = 0
    n_out = 0
    with open(args.edges_in) as fin, open(args.edges_out, "w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            s = rec["subject"]
            o = rec["object"]
            if s in nonhuman or o in nonhuman:
                continue
            fout.write(line)
            surviving_nodes.add(s)
            surviving_nodes.add(o)
            n_out += 1
            if n_in % 1_000_000 == 0:
                log(f"  edges read: {n_in:,}  written: {n_out:,}  "
                    f"({time()-t:.1f}s)")
    log(f"  input edges: {n_in:,}")
    log(f"  edges dropped (touch non-human gene): {n_in - n_out:,}")
    log(f"  edges written: {n_out:,}  -->  {args.edges_out}")
    log(f"  nodes with >=1 surviving edge: {len(surviving_nodes):,}")
    log(f"  ({time()-t:.1f}s)")

    # =============================================================
    # PHASE C: stream nodes again, write non-removed nodes that survive
    # =============================================================
    # A node is written iff it was not flagged non-human AND it appears in at
    # least one surviving edge (isolated-node sweep).
    log("Phase C: writing nodes (drop non-human + isolated)...")
    t = time()
    n_in = 0
    n_out = 0
    n_iso = 0
    with open(args.nodes_in) as fin, open(args.nodes_out, "w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            nid = rec["id"]
            if nid in nonhuman:
                continue
            if nid not in surviving_nodes:
                n_iso += 1
                continue
            fout.write(line)
            n_out += 1
    log(f"  nodes dropped (newly isolated): {n_iso:,}")
    log(f"  nodes written: {n_out:,}  -->  {args.nodes_out}")
    log(f"  ({time()-t:.1f}s)")
    log("DONE")


if __name__ == "__main__":
    main()
