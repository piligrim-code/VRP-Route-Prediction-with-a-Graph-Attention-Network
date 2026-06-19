"""
Enhanced training with route cost in loss function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RouteCostLoss(nn.Module):
    """
    Loss function that combines cross-entropy with route cost.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Weight for route cost loss (0.0 = only CE, 1.0 = only cost)
        """
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            logits: [batch_size, seq_len, num_nodes]
            predictions: [batch_size, seq_len]
            targets: [batch_size, seq_len]
            edge_attr: [num_edges, 3] - [distance, duration, toll_cost]
            edge_index: [2, num_edges]
            batch: [num_nodes]

        Returns:
            loss: Combined loss
            metrics: Dict with loss components
        """
        batch_size, seq_len, num_nodes = logits.shape

        # Cross-entropy loss
        ce_loss = self.ce_loss(
            logits.reshape(-1, num_nodes),
            targets.reshape(-1)
        )

        # Route cost loss (only if alpha > 0)
        if self.alpha > 0:
            route_cost_loss = self._compute_route_cost_loss(
                predictions, targets, edge_attr, edge_index, batch
            )
        else:
            route_cost_loss = torch.tensor(0.0, device=logits.device)

        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * route_cost_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'route_cost_loss': route_cost_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def _compute_route_cost_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute route cost loss based on predicted routes.

        Penalizes routes that are longer than ground truth.
        """
        batch_size = predictions.shape[0]
        device = predictions.device

        # Build edge cost matrix
        # batch contains graph assignment for each node, so len(batch) = total nodes
        num_nodes = len(batch)
        cost_matrix = torch.zeros(num_nodes, num_nodes, device=device)

        # Fill cost matrix with distances
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            distance = edge_attr[i, 0]  # distance_km
            cost_matrix[src, dst] = distance

        total_cost_loss = 0.0
        valid_samples = 0

        # Compute cost for each sample in batch
        for b in range(batch_size):
            pred_route = predictions[b]
            target_route = targets[b]

            # Filter out padding (-1)
            valid_pred = pred_route[pred_route != -1]
            valid_target = target_route[target_route != -1]

            if len(valid_pred) < 2 or len(valid_target) < 2:
                continue

            # Compute predicted route cost
            pred_cost = 0.0
            for i in range(len(valid_pred) - 1):
                src = valid_pred[i].item()
                dst = valid_pred[i + 1].item()
                if src < num_nodes and dst < num_nodes:
                    pred_cost += cost_matrix[src, dst]

            # Compute target route cost
            target_cost = 0.0
            for i in range(len(valid_target) - 1):
                src = valid_target[i].item()
                dst = valid_target[i + 1].item()
                if src < num_nodes and dst < num_nodes:
                    target_cost += cost_matrix[src, dst]

            # Loss: penalize if predicted cost > target cost
            if target_cost > 0:
                cost_ratio = pred_cost / (target_cost + 1e-6)
                total_cost_loss += F.relu(cost_ratio - 1.0)  # Only penalize if worse
                valid_samples += 1

        if valid_samples > 0:
            return total_cost_loss / valid_samples
        else:
            return torch.tensor(0.0, device=device)

