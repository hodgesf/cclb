from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False,
                 num_beam=10, path_topk=10):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
        # add self loop
        # convert homogeneous graphs to knowledge graphs with 1 relation
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        return graph

    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        step_graphs = output["step_graphs"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = [graph.edge_weight for graph in step_graphs]
        edge_grads = autograd.grad(score, edge_weights)
        for graph, edge_grad in zip(step_graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        distances, back_edges = self.beam_search_distance(step_graphs, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for graph in graphs:
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = torch.repeat_interleave(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths
    
@R.register("model.MaskedNBFNet")
class MaskedNBFNet(NeuralBellmanFordNetwork):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            num_relation=None,
            num_entities=None,
            selector_dim=32,
            lambda_l1=1.0e-4,
            audit=False,
            **kwargs):

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_relation=num_relation,
            **kwargs
        )
        relations = num_relation * 2
        self.selector = EdgeSelector(num_entities, relations, selector_dim)
        self.num_hops = len(hidden_dims) # L
        self.lambda_l1 = lambda_l1  # sparsity weight on mask.sum(); controls budget implicitly
        self.audit = audit
        assert not self.symmetric


    def extract_subgraph(self, graph, h):
        """
        This method computes a subgraph that are all within self.num_hops (L) of the head node of the query from 
        the full graph. Runs vectorized BFS (pretty much frontier expansion, I believe) over graph.edge_list
        Args: 
            - graph (torchdrug.data.Graph): the graph object created by the tasks forward pass 
                (passed in as task.fact_graph). 
                - graph.edge_list (LongTensor, (|E|, 3)): the rows of the graph [node_in, node_out, relation]
                - graph.num_node (int): number of nodes in the fact graph 
                - graph.edge_weight (Tensor, |E|): per edge weights
            - h (LongTensor scalar): head node of the query (single query, B=1)
        Returns: 
            - sub_edge_id (LongTensor, [num_sub_edges]): indexes into graph.edge_list of edges whose source is within 
                L-1 hops of h. This becomes the EdgeSelectors candidates  
            - hop_dist (FloatTensor, [N]): shortest path hop distance from h to each node. float('inf') for unreached. 

            AFTER TESTING THIS METHOD IS NOT NEEDED. TURNS OUT L=6 LEADS TO ~99% OF THE GRAPH BEING EXPLORED. Leaving this incase we want to do hops later. 
        """
        node_in, node_out, relation = graph.edge_list.t() #each shape |E|
        N = graph.num_node 
        device = graph.edge_list.device 

        hop_dist = torch.full((N,), float('inf'), device=device) # tensor of length N full of inf 
        visited = torch.zeros((N,), dtype=torch.bool, device=device) # tensor of length N full of booleans 
        hop_dist[h] = 0
        visited[h] = True 

        for el in range(1, self.num_hops + 1): 
            active = visited[node_in] # |E| bool: edges whose source is reached 
            reached = node_out[active] 
            new = reached[~visited[reached]]
            new = torch.unique(new)
            if new.numel() == 0: 
                break
            hop_dist[new] = el 
            visited[new] = True 

        src_dist = hop_dist[node_in]
        keep = src_dist <= self.num_hops - 1
        sub_edge_id = torch.nonzero(keep).squeeze(-1) # long tensor of edge indexes

        return sub_edge_id, hop_dist 

    def score_edges(self, graph, h, r_query):
        """Compute one selector logit per edge in the (augmented) graph for query (h, r_query).

        Args:
            graph (torchdrug.data.Graph): the inverse-augmented graph (output of
                graph.undirected(add_inverse=True) inside forward).
            h (LongTensor, scalar): head node id of the query (B=1).
            r_query (LongTensor, scalar): query relation id (B=1).

        Returns:
            logits (Tensor, |E|): one real logit per edge.
                sigmoid(logit) > 0.5 -> edge included. Consumed by select_mask.
        """
        u, v, r_e = graph.edge_list.t()
        logits = self.selector.score(h, r_query, u, r_e, v)
        return logits

    def select_mask(self, logits):
        """Per-edge Bernoulli-style mask with threshold @ 0.5 and STE.

        Each edge decides independently whether to be included based on whether
        sigmoid(logit) crosses 0.5. Number of selected edges is variable per query;
        sparsity is controlled implicitly by the L1 penalty on mask.sum() in forward.

        Train:  hard {0,1} via threshold; gradient flows through sigmoid via STE.
        Eval:   hard {0,1} via threshold; no STE wrapping needed.

        Args:
            logits (Tensor, [E]): one logit per edge.

        Returns:
            mask (Tensor, [E]) in {0,1}: hard mask in forward; STE-differentiable
                during training (gradient = sigmoid'(logits) for every edge).
        """
        soft = torch.sigmoid(logits)
        mask_hard = (soft > 0.5).float()

        if self.training:
            mask = (mask_hard - soft).detach() + soft
        else:
            mask = mask_hard

        return mask

    def apply_mask(self, graph, mask):
        """Multiply edge_weight by the STE mask. KEEP ALL EDGES.

        Forward: unselected edges have edge_weight * 0 = 0, so they contribute
        nothing to BF messages -- predictor's effective input is the K selected
        edges (faithful hard mask). Same identity as drop-and-rebuild in the
        forward direction.

        Backward: every edge's mask value sits in the autograd graph because
        it multiplies edge_weight. STE routes gradient from mask back to
        sigmoid(scores) for EVERY edge, not just the K selected. Unselected
        edges' selector logits therefore receive a counterfactual gradient --
        "if you had picked me, here is how I would have changed the loss" --
        which is the signal drop-and-rebuild discarded.

        Same code path in train and eval; the only difference is whether the
        incoming mask is STE-wrapped (train) or plain 0/1 (eval).
        """
        new_edge_weight = graph.edge_weight * mask
        return type(graph)(
            graph.edge_list,
            edge_weight=new_edge_weight,
            num_node=graph.num_node,
            num_relation=graph.num_relation,
            meta_dict=graph.meta_dict,
            **graph.data_dict,
        )


    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        """Per-query masked BF: select edges, build a per-query masked graph, run the
        inherited bellmanford, gather tail scores. B=1 looped so each
        query gets its own subgraph. Reuses parent helpers (remove_easy_edges,
        undirected, negative_sample_to_tail, bellmanford, self.mlp)

        Args:
            graph (torchdrug.data.Graph): task.fact_graph
            h_index, t_index, r_index (LongTensor [B, 1+num_neg]): per-row positives
                + negatives. Shape preserved on return.
            all_loss: training-mode flag the parent uses to decide remove_easy_edges.
            metric: passed through, unused here.

        Returns:
            score (Tensor [B, 1+num_neg]): one logit per (query, candidate) pair.
        """
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        assert graph.num_relation, "MaskedNBFNet requires a relational graph"
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        if self.audit:
            self._audit_buffer = []

        B = h_index.shape[0]
        scores = []
        for b in range(B):
            h_b = h_index[b, 0]
            r_b = r_index[b, 0]
            t_b = t_index[b]
            logits = self.score_edges(graph, h_b, r_b)
            mask   = self.select_mask(logits)
            g_b    = self.apply_mask(graph, mask)
            out = self.bellmanford(g_b, h_b.unsqueeze(0), r_b.unsqueeze(0))
            feat = out["node_feature"].squeeze(1)
            feat_at_t = feat[t_b]
            score_b   = self.mlp(feat_at_t).squeeze(-1)
            scores.append(score_b)
            # L1 sparsity penalty: without this, selector trivially picks all edges
            # since unconstrained sigmoids saturate high under BCE pressure.
            if all_loss is not None and self.training:
                all_loss += self.lambda_l1 * mask.sum()
            if self.audit:
                self._audit_buffer.append({
                    "query":             (h_b.item(), r_b.item()),
                    "selected_edge_ids": mask.detach().nonzero().squeeze(-1).cpu(),
                    "k_used":            int(mask.detach().sum().item()),
                    "logit_mean":        logits.detach().mean().item(),
                    "logit_std":         logits.detach().std().item(),
                })
        if metric is not None and self.audit and len(self._audit_buffer) > 0:
            ks = [e["k_used"] for e in self._audit_buffer]
            n_edges = max(1, int(graph.num_edge))
            device = h_index.device
            metric["k_used_mean"] = torch.tensor(sum(ks) / len(ks), device=device)
            metric["k_used_min"] = torch.tensor(float(min(ks)), device=device)
            metric["k_used_max"] = torch.tensor(float(max(ks)), device=device)
            metric["mask_density"] = torch.tensor((sum(ks) / len(ks)) / n_edges, device=device)
            metric["logit_mean_avg"] = torch.tensor(sum(e["logit_mean"] for e in self._audit_buffer) / len(self._audit_buffer),
                  device=device,
              )
        return torch.stack(scores, dim=0).view(shape)    



    def get_audit(self):
        """Return the audit buffer populated by the most recent forward() call.

      Returns:
          list of dict: one entry per query in the last forward's batch.
              Each entry contains:
                - "query":              (h_id, r_id); the query this entry is for
                - "selected_edge_ids":  LongTensor on CPU; indices where mask==1
                - "k_used":             int; |selected| this query (variable per query)
                - "logit_mean":         float; mean of selector logits over all edges
                - "logit_std":          float; stdev of selector logits over all edges
          Empty list if self.audit is False or forward() hasn't been called yet.
        """
        return list(getattr(self, "_audit_buffer", []))
    
@R.register("model.EdgeSelector")
class EdgeSelector(nn.Module):
    def __init__(self, num_entities, num_relations, dim,
                   num_mlp_layer=2):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim) # num_relations = 2*|R|
        # cold-start defense: small-variance init -> near-uniform initial logits
        # near zero -> sigmoid(0) = 0.5 -> threshold straddles the boundary so
        # small score perturbations flip edges in/out (high gradient sensitivity).
        nn.init.normal_(self.entity.weight, std=0.01)
        nn.init.normal_(self.relation.weight, std=0.01)
        feat_dim = 5*dim # [h, r_q, u, r_e, v]
        self.mlp = layers.MLP(feat_dim, [feat_dim] * (num_mlp_layer-1) + [1])

    def score(self, h, r_query, u, r_e, v):
        """Per-edge logits for one query (B=1).

        Looks up h, r_query, u, r_e, v in the selector's embedding table
        builds the per-edge feature [E[h], R[r_q], E[u], R[r_e], E[v]], and runs
        the MLP. Returns one real-valued logit per edge.

        Args:
            h (LongTensor, scalar): query head node id.
            r_query (LongTensor, scalar): query relation id.
            u (LongTensor, [E]): source node id of each edge (node_in).
            r_e (LongTensor, [E]): relation id of each edge.
            v (LongTensor, [E]): destination node id of each edge (node_out).

        Returns:
            logits (Tensor, [E]): one real logit per edge.
        """
        E = u.shape[0]
        e_h   = self.entity(h)
        e_r_q = self.relation(r_query)
        e_u   = self.entity(u)
        e_v   = self.entity(v)
        e_r_e = self.relation(r_e)


        e_h_b   = e_h.unsqueeze(0).expand(E, -1)
        e_r_q_b = e_r_q.unsqueeze(0).expand(E, -1)

        feat = torch.cat([e_h_b, e_r_q_b, e_u, e_r_e, e_v], dim=-1)   # (E, 5*dim)

        return self.mlp(feat).squeeze(-1)

