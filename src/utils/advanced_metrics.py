"""
Metrics for evaluating predicted route quality.
"""
import torch
from typing import List, Dict


def position_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -1) -> float:
    """Fraction of correctly predicted nodes across all valid positions."""
    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask
    total_correct = correct.sum().item()
    total_positions = valid_mask.sum().item()
    return total_correct / total_positions if total_positions > 0 else 0.0


def sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -1) -> float:
    """Exact sequence match: a route counts only if every node matches."""
    batch_size = predictions.shape[0]
    correct_sequences = 0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        pred_valid = pred[valid_mask]
        tgt_valid = tgt[valid_mask]

        if len(pred_valid) == len(tgt_valid) and torch.all(pred_valid == tgt_valid):
            correct_sequences += 1

    return correct_sequences / batch_size if batch_size > 0 else 0.0


def prefix_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                    k: int = 5, ignore_index: int = -1) -> float:
    """Accuracy over the first k positions of each route."""
    batch_size = predictions.shape[0]
    total_correct = 0
    total_positions = 0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        actual_k = min(k, len(valid_indices))
        for j in range(actual_k):
            idx = valid_indices[j]
            if pred[idx] == tgt[idx]:
                total_correct += 1
            total_positions += 1

    return total_correct / total_positions if total_positions > 0 else 0.0


def edit_distance(pred_seq: List[int], target_seq: List[int]) -> int:
    """Levenshtein distance between two sequences."""
    n, m = len(pred_seq), len(target_seq)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == target_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1]  # substitution
                )

    return dp[n][m]


def normalized_edit_distance(predictions: torch.Tensor, targets: torch.Tensor,
                             ignore_index: int = -1) -> float:
    """Edit distance normalized by target length, averaged over the batch.

    Ranges from 0 (identical) to 1 (completely different).
    """
    batch_size = predictions.shape[0]
    total_distance = 0.0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        pred_valid = pred[valid_mask].cpu().tolist()
        tgt_valid = tgt[valid_mask].cpu().tolist()

        if len(tgt_valid) == 0:
            continue

        dist = edit_distance(pred_valid, tgt_valid)
        total_distance += dist / len(tgt_valid)

    return total_distance / batch_size if batch_size > 0 else 0.0


def longest_common_subsequence(pred_seq: List[int], target_seq: List[int]) -> int:
    """Length of the longest common subsequence."""
    n, m = len(pred_seq), len(target_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == target_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def lcs_ratio(predictions: torch.Tensor, targets: torch.Tensor,
              ignore_index: int = -1) -> float:
    """LCS length divided by target length, averaged over the batch.

    Indicates the fraction of nodes present in the correct order.
    """
    batch_size = predictions.shape[0]
    total_ratio = 0.0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        pred_valid = pred[valid_mask].cpu().tolist()
        tgt_valid = tgt[valid_mask].cpu().tolist()

        if len(tgt_valid) == 0:
            continue

        lcs_len = longest_common_subsequence(pred_valid, tgt_valid)
        total_ratio += lcs_len / len(tgt_valid)

    return total_ratio / batch_size if batch_size > 0 else 0.0


def first_error_position(predictions: torch.Tensor, targets: torch.Tensor,
                         ignore_index: int = -1) -> float:
    """Average position of the first error, normalized by route length.

    Indicates how far the model predicts correctly before diverging.
    """
    batch_size = predictions.shape[0]
    total_position = 0.0
    count = 0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        first_error = len(valid_indices)  # default: no errors
        for j, idx in enumerate(valid_indices):
            if pred[idx] != tgt[idx]:
                first_error = j
                break

        total_position += first_error / len(valid_indices)
        count += 1

    return total_position / count if count > 0 else 0.0


def node_coverage(predictions: torch.Tensor, targets: torch.Tensor,
                  ignore_index: int = -1) -> float:
    """Fraction of ground-truth nodes present in the prediction, ignoring order."""
    batch_size = predictions.shape[0]
    total_coverage = 0.0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        valid_mask = tgt != ignore_index
        pred_valid = set(pred[valid_mask].cpu().tolist())
        tgt_valid = set(tgt[valid_mask].cpu().tolist())

        if len(tgt_valid) == 0:
            continue

        intersection = pred_valid & tgt_valid
        total_coverage += len(intersection) / len(tgt_valid)

    return total_coverage / batch_size if batch_size > 0 else 0.0


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                        ignore_index: int = -1) -> Dict[str, float]:
    """Compute all metrics at once."""
    return {
        'position_accuracy': position_accuracy(predictions, targets, ignore_index),
        'sequence_accuracy': sequence_accuracy(predictions, targets, ignore_index),
        'prefix_3_accuracy': prefix_accuracy(predictions, targets, k=3, ignore_index=ignore_index),
        'prefix_5_accuracy': prefix_accuracy(predictions, targets, k=5, ignore_index=ignore_index),
        'normalized_edit_distance': normalized_edit_distance(predictions, targets, ignore_index),
        'lcs_ratio': lcs_ratio(predictions, targets, ignore_index),
        'first_error_position': first_error_position(predictions, targets, ignore_index),
        'node_coverage': node_coverage(predictions, targets, ignore_index),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a readable table."""
    print(f"\n{'='*60}")
    print(f"{prefix} Metrics:")
    print(f"{'='*60}")
    print(f"Position Accuracy:        {metrics['position_accuracy']*100:.2f}%")
    print(f"Sequence Accuracy:        {metrics['sequence_accuracy']*100:.2f}%")
    print(f"Prefix-3 Accuracy:        {metrics['prefix_3_accuracy']*100:.2f}%")
    print(f"Prefix-5 Accuracy:        {metrics['prefix_5_accuracy']*100:.2f}%")
    print(f"Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f}")
    print(f"LCS Ratio:                {metrics['lcs_ratio']*100:.2f}%")
    print(f"First Error Position:     {metrics['first_error_position']*100:.2f}%")
    print(f"Node Coverage:            {metrics['node_coverage']*100:.2f}%")
    print(f"{'='*60}\n")
