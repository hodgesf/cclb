"""End-to-end smoke test for MaskedNBFNet.

Builds a tiny toy graph, instantiates MaskedNBFNet + EdgeSelector with small
dims, runs forward + backward, and asserts:
  1. No exceptions (catches the apply_mask torchdrug context-manager risk).
  2. Selector entity embedding gradients are non-None and have nonzero values
     (straight-through gradient reaches the selector).
  3. Selector relation embedding gradients are non-None and have nonzero values.
  4. Selector MLP first-layer weights have gradient.
  5. Predictor (parent NBFNet) layer weights have gradient (still trains).
  6. get_audit() returns one entry per query with the expected dict shape.

Runs CPU-only in a few seconds. This is the gate before any real-data run.
"""
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NBFNet"))

from torchdrug import data
from nbfnet.model import MaskedNBFNet  # noqa: E402


def build_toy_graph():
    """Tiny relational graph: 12 nodes, 3 relations, ~30 directed edges."""
    edges = [
        # (head, tail, relation)
        (0, 1, 0), (0, 2, 0), (0, 3, 1),
        (1, 4, 0), (1, 5, 1),
        (2, 5, 0), (2, 6, 2),
        (3, 6, 1), (3, 7, 0),
        (4, 7, 2), (4, 8, 0),
        (5, 8, 1), (5, 9, 0),
        (6, 9, 2), (6, 10, 0),
        (7, 10, 1), (7, 11, 2),
        (8, 11, 0), (8, 0, 1),
        (9, 0, 2), (9, 1, 0),
        (10, 1, 1), (10, 2, 2),
        (11, 2, 0), (11, 3, 1),
        (0, 4, 2), (1, 7, 1),
        (2, 8, 1), (3, 9, 2),
        (4, 10, 1),
    ]
    edge_list = torch.tensor(edges, dtype=torch.long)
    return data.Graph(edge_list, num_node=12, num_relation=3)


def main():
    torch.manual_seed(0)
    graph = build_toy_graph()
    print(f"toy graph: |V|={graph.num_node}  |E|={graph.num_edge}  |R|={graph.num_relation}")

    # Tiny config; lambda_l1=0 so the smoke test doesn't fight per-edge sigmoid.
    # With small-variance init, logits ~ 0 -> sigmoid ~ 0.5 -> threshold straddles
    # the boundary, so ~half of the edges get selected initially. We just want
    # gradient flow to work end-to-end.
    model = MaskedNBFNet(
        input_dim=8,
        hidden_dims=[8, 8, 8],
        num_relation=3,
        num_entities=12,
        selector_dim=8,
        lambda_l1=0.0,
        audit=True,
    )
    model.train()

    # One batch: 2 queries, 1 positive + 3 candidate tails each.
    # Shape (B=2, 1+num_neg=4).
    h_index = torch.tensor([[0, 0, 0, 0], [5, 5, 5, 5]], dtype=torch.long)
    t_index = torch.tensor([[1, 2, 3, 4], [8, 9, 0, 1]], dtype=torch.long)
    r_index = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long)

    # ---- forward ----
    score = model(graph, h_index, t_index, r_index, all_loss=torch.tensor(0.0))
    print(f"score shape: {tuple(score.shape)}  expected: (2, 4)")
    assert score.shape == (2, 4), f"score shape mismatch: {score.shape}"

    # ---- backward ----
    # Simple loss: positive at column 0 should outrank negatives. CE-style.
    target = torch.zeros(2, dtype=torch.long)  # column 0 is the positive
    loss = torch.nn.functional.cross_entropy(score, target)
    print(f"loss: {loss.item():.4f}")
    loss.backward()

    # ---- assertions ----
    print("\n--- gradient flow checks ---")
    sel_ent_grad = model.selector.entity.weight.grad
    assert sel_ent_grad is not None, "selector.entity.weight.grad is None (STE broken)"
    assert (sel_ent_grad != 0).any(), "selector.entity.weight.grad is all zeros (STE broken)"
    print(f"  selector.entity.weight.grad : nnz={(sel_ent_grad != 0).sum().item()}  norm={sel_ent_grad.norm().item():.4e}")

    sel_rel_grad = model.selector.relation.weight.grad
    assert sel_rel_grad is not None, "selector.relation.weight.grad is None"
    assert (sel_rel_grad != 0).any(), "selector.relation.weight.grad is all zeros"
    print(f"  selector.relation.weight.grad: nnz={(sel_rel_grad != 0).sum().item()}  norm={sel_rel_grad.norm().item():.4e}")

    # torchdrug.layers.MLP stores Linear layers in self.layers
    mlp_first_linear = model.selector.mlp.layers[0]
    assert mlp_first_linear.weight.grad is not None, "selector.mlp first layer has no grad"
    assert (mlp_first_linear.weight.grad != 0).any(), "selector.mlp first layer grad is zero"
    print(f"  selector.mlp.layers[0].weight.grad : norm={mlp_first_linear.weight.grad.norm().item():.4e}")

    # predictor (parent NBFNet) still trains
    pred_first_layer = model.layers[0]
    has_predictor_grad = any(
        p.grad is not None and (p.grad != 0).any() for p in pred_first_layer.parameters()
    )
    assert has_predictor_grad, "predictor first BF layer has no gradient (something broke parent)"
    print(f"  model.layers[0] (predictor BF): gradient present")

    # mlp head (parent's scoring head)
    assert model.mlp.layers[0].weight.grad is not None, "parent scoring mlp has no grad"
    print(f"  model.mlp (predictor head) : gradient present")

    # ---- audit ----
    print("\n--- audit check ---")
    audit = model.get_audit()
    assert len(audit) == 2, f"expected 2 audit entries, got {len(audit)}"
    for i, entry in enumerate(audit):
        assert "query" in entry and "selected_edge_ids" in entry and "k_used" in entry
        assert "logit_mean" in entry and "logit_std" in entry
        # variable subset size now; just verify the count matches k_used and is in range
        assert entry["selected_edge_ids"].numel() == entry["k_used"], \
            f"query {i}: |selected_edge_ids| != k_used"
        print(f"  query {i}: query={entry['query']}  selected={entry['selected_edge_ids'].numel()} edges  "
              f"k_used={entry['k_used']}  logit_mean={entry['logit_mean']:.3f}  logit_std={entry['logit_std']:.3f}")

    print("\nALL CHECKS PASSED -- masked forward+backward works end-to-end on toy graph.")


if __name__ == "__main__":
    main()
