"""Cross-entropy plus an independent-marginal transition-cost surrogate.

The surrogate is differentiable in logits, not a claim of a valid VRP route
or an exact expected cost under the autoregressive policy. Missing edges
(including self-transitions) receive a penalty above all observed edge costs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RouteCostLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be between zero and one')
        self.alpha = alpha

    def forward(self, logits, predictions, targets, edge_attr, edge_index, batch,
                raw_edge_costs=None):
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError('logits and targets must have matching batch/sequence dimensions')
        if targets.dtype != torch.long or batch.dtype != torch.long or edge_index.dtype != torch.long:
            raise ValueError('targets, batch and edge_index must contain int64 indices')
        if batch.ndim != 1 or not torch.equal(batch.unique(sorted=True), torch.arange(logits.size(0), device=batch.device)):
            raise ValueError('batch must identify every graph from zero to batch_size-1')
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError('edge_index must have shape [2, edges]')
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= batch.numel()):
            raise ValueError('Edge endpoint out of range')
        if edge_index.numel() and (batch[edge_index[0]] != batch[edge_index[1]]).any():
            raise ValueError('Edges must not cross graphs')
        if self.alpha > 0:
            if raw_edge_costs is None:
                raise ValueError('raw_edge_costs are required; normalized edge_attr is not a distance')
            if raw_edge_costs.shape != (edge_index.size(1),) or not torch.isfinite(raw_edge_costs).all() or (raw_edge_costs < 0).any():
                raise ValueError('raw_edge_costs must be finite nonnegative distances, one per edge')

        zero = torch.where(torch.isfinite(logits), logits, 0).sum() * 0
        ce_sum, valid_count, penalties = zero, 0, []
        for graph in range(logits.size(0)):
            nodes = (batch == graph).nonzero(as_tuple=True)[0]
            n = nodes.numel()
            if n > logits.size(-1):
                raise ValueError('logits do not cover all graph nodes')
            y = targets[graph]
            valid = y != -1
            if ((y < -1) | (y >= n)).any():
                raise ValueError('Target node is outside its graph')
            if not valid.any():
                continue
            scores = logits[graph, :, :n]
            active = scores[valid]
            if torch.isnan(active).any() or torch.isposinf(active).any() or not torch.isfinite(active).any(dim=-1).all():
                raise ValueError('Every active position must have a finite node score')
            ce_sum = ce_sum + F.cross_entropy(active, y[valid], reduction='sum')
            valid_count += int(valid.sum())
            transitions = valid[:-1] & valid[1:]
            if self.alpha == 0 or not transitions.any():
                continue

            # Convert globally batched edge endpoints to graph-local indices.
            local = torch.full_like(batch, -1)
            local[nodes] = torch.arange(n, device=batch.device)
            edges = batch[edge_index[0]] == graph
            costs = raw_edge_costs[edges].to(dtype=logits.dtype)
            scale = costs.max().clamp_min(1) if costs.numel() else logits.new_tensor(1)
            missing_cost = scale * 2 + 1
            matrix = missing_cost.expand(n, n).clone()
            endpoints = local[edge_index[:, edges]]
            for i in range(costs.numel()):
                src, dst = endpoints[:, i]
                matrix[src, dst] = torch.minimum(matrix[src, dst], costs[i])
            safe_scores = scores.masked_fill(~valid.unsqueeze(-1), 0)
            probs = torch.softmax(safe_scores, dim=-1)
            expected = ((probs[:-1] @ matrix) * probs[1:]).sum(dim=-1)
            penalties.append((expected[transitions] / scale).mean())

        ce = ce_sum / max(valid_count, 1)
        route = torch.stack(penalties).mean() if penalties else zero
        total = (1 - self.alpha) * ce + self.alpha * route
        return total, {'ce_loss': ce.detach().item(),
                       'route_cost_loss': route.detach().item(),
                       'total_loss': total.detach().item()}
