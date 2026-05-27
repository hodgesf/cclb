"""Analyze the raw RDKG dumps in cclb/data/.

Streams `all_nodes.jsonl` and `all_edges.jsonl` once each and prints
structured descriptive statistics:

  NODES
    counts by biological bucket (Gene/Disease/Drug/GO/Pathway/Anatomy/Other)
    counts by primary biolink category
    counts by CURIE prefix (which source DB the identifier came from)

  EDGES
    predicate distribution
    biolink association-class distribution
    primary knowledge source distribution (which DB produced the edge)
    agent_type distribution (manual vs automated)
    knowledge_level distribution (assertion vs inference vs lookup ...)
    self-loop and publication-supported counts

  STRUCTURE
    degree histogram across bins
    top hub nodes
    isolated node count

  METAEDGES
    counts of (subject_bucket, predicate, object_bucket) triples
    helps spot which type-level edge patterns dominate the graph

  GENE/DISEASE FOCUS
    direct gene<->disease edge counts
    predicates and primary sources that produce them

Usage:
  python analyze_rdkg.py --nodes data/all_nodes.jsonl \\
                         --edges data/all_edges.jsonl \\
                         [--top N]                  # default 20

Output goes to stdout. Progress is logged to stderr.
"""

import argparse
import json
import sys
from collections import Counter
from time import time


# ---------- helpers ----------------------------------------------------------


def curie_prefix(curie):
    """Return the part of a CURIE before the first ':', or the whole string."""
    i = curie.find(":")
    return curie[:i] if i > 0 else curie


def primary_category(record):
    """The first item in a biolink `category` list — the most-specific class."""
    cats = record.get("category") or []
    if isinstance(cats, str):
        return cats
    return cats[0] if cats else "<no category>"


def bucket(category_list):
    """Map a node's full biolink category list into one coarse bucket.

    Order matters: we check the more specific labels first so a node
    typed both as Gene and ChemicalEntity falls into Gene/Protein.
    """
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
               "biolink:Cell", "biolink:CellularComponent"}:
        return "Anatomy"
    if "biolink:Publication" in cats:
        return "Publication"
    return "Other"


def primary_source(rec):
    """Return the resource_id of the edge's primary_knowledge_source, or fall
    back to the first source listed, or '<no_source>' if none.
    """
    sources = rec.get("sources") or []
    for s in sources:
        if s.get("resource_role") == "primary_knowledge_source":
            return s.get("resource_id", "<no_id>")
    if sources:
        return sources[0].get("resource_id", "<no_id>")
    return "<no_source>"


def print_header(title):
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def print_counter(label, counter, top, total=None):
    """Print the top-N items of a Counter as a percentage of the total
    (defaults to sum(counter)).
    """
    print(f"\n{label}:")
    total = total if total is not None else sum(counter.values())
    if total == 0:
        print("  (empty)")
        return
    listed = 0
    for k, v in counter.most_common(top):
        listed += v
        print(f"  {v:>12,d}  {100*v/total:6.2f}%  {k}")
    rest = total - listed
    n_rest_categories = max(0, len(counter) - top)
    if rest > 0:
        print(f"  {rest:>12,d}  {100*rest/total:6.2f}%  "
              f"(rest of {n_rest_categories:,} categories)")


# ---------- pass 1: nodes ----------------------------------------------------


def analyze_nodes(path):
    """Stream the node file. Returns the per-node bucket map and three counters.

    node_bucket is the only large object that we keep around (used by the
    edge pass to type each endpoint). Everything else is a Counter, cheap.
    """
    node_bucket = {}
    primary_cat = Counter()
    bucket_counts = Counter()
    prefix_counts = Counter()
    has_description = 0
    n = 0
    t0 = time()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            nid = rec["id"]
            cat = rec.get("category") or []
            b = bucket(cat)
            node_bucket[nid] = b
            primary_cat[primary_category(rec)] += 1
            bucket_counts[b] += 1
            prefix_counts[curie_prefix(nid)] += 1
            if rec.get("description"):
                has_description += 1
            n += 1
            if n % 200000 == 0:
                print(f"  nodes: {n:,} ({time()-t0:.1f}s)", file=sys.stderr)
    return {
        "node_bucket": node_bucket,
        "primary_cat": primary_cat,
        "bucket_counts": bucket_counts,
        "prefix_counts": prefix_counts,
        "n": n,
        "has_description": has_description,
    }


# ---------- pass 2: edges ----------------------------------------------------


def analyze_edges(path, node_bucket):
    """Stream the edge file. Counts predicate, edge category, primary source,
    agent_type, knowledge_level, metaedge type signatures, and per-node degree.
    Records self-loops and publication-supported edges.
    """
    pred = Counter()
    edge_cat = Counter()
    src = Counter()
    agent = Counter()
    klevel = Counter()
    metaedge = Counter()
    degree = Counter()
    self_loops = 0
    pub_supported = 0
    n = 0
    t0 = time()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            s = rec["subject"]
            o = rec["object"]
            p = rec.get("predicate", "<missing>")
            pred[p] += 1
            c = rec.get("category") or []
            edge_cat[c[0] if isinstance(c, list) and c else (c if isinstance(c, str) else "<none>")] += 1
            src[primary_source(rec)] += 1
            agent[rec.get("agent_type", "<missing>")] += 1
            klevel[rec.get("knowledge_level", "<missing>")] += 1
            degree[s] += 1
            degree[o] += 1
            if s == o:
                self_loops += 1
            sb = node_bucket.get(s, "Unknown")
            ob = node_bucket.get(o, "Unknown")
            metaedge[(sb, p, ob)] += 1
            if rec.get("publications"):
                pub_supported += 1
            n += 1
            if n % 1000000 == 0:
                print(f"  edges: {n:,} ({time()-t0:.1f}s)", file=sys.stderr)
    return {
        "pred": pred,
        "edge_cat": edge_cat,
        "src": src,
        "agent": agent,
        "klevel": klevel,
        "metaedge": metaedge,
        "degree": degree,
        "self_loops": self_loops,
        "pub_supported": pub_supported,
        "n": n,
    }


# ---------- reports ----------------------------------------------------------


def report_nodes(nd, top):
    print_header("NODES")
    print(f"  total nodes: {nd['n']:,}")
    print(f"  with description: {nd['has_description']:,} "
          f"({100*nd['has_description']/nd['n']:.1f}%)")
    print_counter("by biological bucket", nd["bucket_counts"], top, total=nd["n"])
    print_counter(f"by primary biolink category (top {top})", nd["primary_cat"], top,
                  total=nd["n"])
    print_counter(f"by CURIE prefix (top {top})", nd["prefix_counts"], top,
                  total=nd["n"])


def report_edges(ed, top):
    print_header("EDGES")
    print(f"  total edges: {ed['n']:,}")
    print(f"  self-loops: {ed['self_loops']:,} "
          f"({100*ed['self_loops']/ed['n']:.3f}%)")
    print(f"  publication-supported: {ed['pub_supported']:,} "
          f"({100*ed['pub_supported']/ed['n']:.1f}%)")
    print_counter(f"predicates (top {top})", ed["pred"], top, total=ed["n"])
    print_counter(f"biolink association class (top {top})", ed["edge_cat"], top,
                  total=ed["n"])
    print_counter(f"primary knowledge source (top {top})", ed["src"], top,
                  total=ed["n"])
    print_counter("agent_type", ed["agent"], top, total=ed["n"])
    print_counter("knowledge_level", ed["klevel"], top, total=ed["n"])


def report_structure(nd, ed, top):
    print_header("STRUCTURE")
    deg = ed["degree"]
    n_with = len(deg)
    n_iso = nd["n"] - n_with
    print(f"  nodes with >=1 edge: {n_with:,}")
    print(f"  isolated nodes (no edges in this file): {n_iso:,}")

    # Degree percentiles
    deg_values = sorted(deg.values())
    if deg_values:
        def pct(p):
            return deg_values[min(len(deg_values) - 1, int(len(deg_values) * p))]
        print(f"\n  degree percentiles:")
        print(f"    min  : {deg_values[0]:,}")
        print(f"    p50  : {pct(0.50):,}")
        print(f"    p90  : {pct(0.90):,}")
        print(f"    p99  : {pct(0.99):,}")
        print(f"    p999 : {pct(0.999):,}")
        print(f"    max  : {deg_values[-1]:,}")

    # Degree histogram (each bin is "<= b" relative to previous bin)
    bins = [1, 2, 5, 10, 50, 100, 500, 1000, 10000, 100000, 1000000]
    bin_counts = Counter()
    prev = 0
    for d in deg_values:
        placed = False
        for b in bins:
            if d <= b:
                bin_counts[b] += 1
                placed = True
                break
        if not placed:
            bin_counts["overflow"] += 1
    print(f"\n  degree histogram (count of nodes in each bin):")
    prev = 0
    for b in bins:
        label = f"({prev:>7,} .. {b:>7,}]" if prev > 0 else f"({prev}, {b:>7,}]"
        print(f"    {label}: {bin_counts[b]:,}")
        prev = b
    print(f"    (> {bins[-1]:,})          : {bin_counts['overflow']:,}")

    # Top hubs (highest-degree nodes)
    print(f"\n  top {top} hub nodes (degree, bucket, CURIE):")
    top_hubs = sorted(deg.items(), key=lambda kv: -kv[1])[:top]
    for nid, d in top_hubs:
        b = nd["node_bucket"].get(nid, "Unknown")
        print(f"    {d:>10,d}  {b:<15s}  {nid}")


def report_metaedges(ed, top):
    print_header("METAEDGES")
    print(f"\n  top {top} (subject_bucket -- predicate --> object_bucket):")
    for (sb, p, ob), c in ed["metaedge"].most_common(top):
        pct = 100.0 * c / ed["n"]
        # Trim biolink: prefix for readability
        p_short = p.replace("biolink:", "")
        print(f"    {c:>12,d}  {pct:6.2f}%  "
              f"{sb:<14s} -- {p_short:<35s} --> {ob}")
    rest_n = sum(c for (k, c) in ed["metaedge"].items()
                 if (k[0], k[1], k[2]) not in
                 {kk for kk, _ in ed["metaedge"].most_common(top)})
    print(f"  {rest_n:>14,d}  {100*rest_n/ed['n']:6.2f}%  "
          f"(rest of {len(ed['metaedge']):,} metaedge types)")


def report_gene_disease(ed, top):
    """Edges that directly connect a Gene/Protein node to a Disease node, in
    either direction. Breaks down by predicate and primary source."""
    print_header("GENE / DISEASE FOCUS")
    gd_total = 0
    gd_pred = Counter()
    for (sb, p, ob), c in ed["metaedge"].items():
        is_gd = ((sb == "Gene/Protein" and ob == "Disease") or
                 (sb == "Disease" and ob == "Gene/Protein"))
        if is_gd:
            gd_total += c
            gd_pred[p] += c
    print(f"\n  direct Gene/Protein <-> Disease edges: {gd_total:,} "
          f"({100*gd_total/ed['n']:.3f}% of all edges)")
    print_counter("predicates used for Gene<->Disease", gd_pred, top, total=gd_total)


# ---------- entry point ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes", required=True, help="path to all_nodes.jsonl")
    parser.add_argument("--edges", required=True, help="path to all_edges.jsonl")
    parser.add_argument("--top", type=int, default=20,
                        help="show top-N per ranking (default 20)")
    args = parser.parse_args()

    print(f"reading nodes from {args.nodes} ...", file=sys.stderr)
    nd = analyze_nodes(args.nodes)
    print(f"reading edges from {args.edges} ...", file=sys.stderr)
    ed = analyze_edges(args.edges, nd["node_bucket"])

    report_nodes(nd, args.top)
    report_edges(ed, args.top)
    report_structure(nd, ed, args.top)
    report_metaedges(ed, args.top)
    report_gene_disease(ed, args.top)

    print_header("DONE")


if __name__ == "__main__":
    main()
