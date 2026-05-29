import torch, random, time

PATH = "splits_human/train.txt"   # adjust to wherever you're running this
L = 6

# --- read train.txt directly (this IS effectively the fact_graph: train edges only) ---
t0 = time.time()
ent, rel = {}, {}
ins, outs, rs = [], [], []
with open(PATH) as f:
    for line in f:
        h, r, t = line.rstrip().split("\t")
        if h not in ent: ent[h] = len(ent)
        if t not in ent: ent[t] = len(ent)
        if r not in rel: rel[r] = len(rel)
        ins.append(ent[h]); outs.append(ent[t]); rs.append(rel[r])
N, R = len(ent), len(rel)
print(f"{len(ins):,} edges, {N:,} entities, {R:,} relations in {time.time()-t0:.1f}s")

# --- augment with inverse edges (what undirected(add_inverse=True) does) ---
ins_t  = torch.tensor(ins, dtype=torch.long)
outs_t = torch.tensor(outs, dtype=torch.long)
rs_t   = torch.tensor(rs,  dtype=torch.long)
node_in  = torch.cat([ins_t,  outs_t])              # forward + inverse sources
node_out = torch.cat([outs_t, ins_t])               # forward + inverse destinations
rel_aug  = torch.cat([rs_t,  rs_t + R])             # inverse relations get id + R
edge_list = torch.stack([node_in, node_out, rel_aug], dim=1)
E = edge_list.shape[0]
print(f"augmented |E|={E:,}  |V|={N:,}")

# --- the BFS (inlined; same body as the method) ---
def extract_subgraph(edge_list, num_node, h, num_hops):
    node_in, node_out, _ = edge_list.t()
    hop = torch.full((num_node,), float("inf"))
    seen = torch.zeros(num_node, dtype=torch.bool)
    hop[h] = 0; seen[h] = True
    for ell in range(1, num_hops + 1):
        reached = node_out[seen[node_in]]
        new = torch.unique(reached[~seen[reached]])
        if new.numel() == 0: break
        hop[new] = ell; seen[new] = True
    keep = hop[node_in] <= num_hops - 1
    return torch.nonzero(keep).squeeze(-1), hop

# --- measure on 20 random heads ---
random.seed(0)
heads = random.sample(range(N), 20)
print(f"\nL={L}  L-hop ball over 20 random heads:")
for h in heads:
    t1 = time.time()
    sub, hop = extract_subgraph(edge_list, N, h, L)
    reached = (hop != float("inf")).sum().item()
    print(f"  h={h:>7}  |sub|={sub.numel():>9} ({sub.numel()/E:5.1%})"
        f"  reached={reached:>7} ({reached/N:5.1%})  [{(time.time()-t1)*1000:.0f}ms]")