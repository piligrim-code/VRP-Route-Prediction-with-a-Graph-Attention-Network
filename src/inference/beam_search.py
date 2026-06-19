"""
Beam Search inference for VRP model.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class BeamSearchVRP:
    """
    Beam Search decoder for VRP model.
    """

    def __init__(self, model, beam_width: int = 5, max_length: int = 50):
        """
        Args:
            model: AttentionVRP model
            beam_width: Number of beams to keep
            max_length: Maximum route length
        """
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length

    def search(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[List[int], float]:
        """
        Perform beam search to find best route.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]

        Returns:
            best_sequence: List of node indices
            best_score: Log probability of best sequence
        """
        device = x.device

        # Encode graph
        node_embeddings = self.model.encode_graph(x, edge_index, edge_attr, batch)

        # Get batch info
        num_graphs = batch.max().item() + 1
        batch_size = num_graphs

        # Organize node embeddings
        node_embeddings_batched = []
        num_nodes_per_graph = []
        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = node_embeddings[mask]
            node_embeddings_batched.append(graph_nodes)
            num_nodes_per_graph.append(graph_nodes.size(0))

        max_nodes = max(num_nodes_per_graph)
        node_embeddings_padded = torch.zeros(
            batch_size, max_nodes, self.model.hidden_dim, device=device
        )
        for i, emb in enumerate(node_embeddings_batched):
            node_embeddings_padded[i, :emb.size(0)] = emb

        # Initialize beams
        # Each beam: (sequence, score, lstm_state, visited_mask)
        beams = []

        # Initial state
        decoder_input = self.model.start_token.expand(1, 1, -1)
        h_0 = torch.zeros(1, 1, self.model.hidden_dim, device=device)
        c_0 = torch.zeros(1, 1, self.model.hidden_dim, device=device)
        lstm_state = (h_0, c_0)

        visited_mask = torch.ones(1, max_nodes, device=device)
        visited_mask[0, num_nodes_per_graph[0]:] = 0

        beams.append(([], 0.0, lstm_state, visited_mask, decoder_input))

        # Beam search loop
        for step in range(min(self.max_length, max_nodes)):
            candidates = []

            for sequence, score, lstm_state, visited_mask, decoder_input in beams:
                # LSTM step
                lstm_out, new_lstm_state = self.model.lstm(decoder_input, lstm_state)
                decoder_state = lstm_out.squeeze(1)

                # Get logits
                logits, _ = self.model.decoder(
                    decoder_state,
                    node_embeddings_padded,
                    mask=visited_mask
                )

                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)

                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(
                    log_probs[0], min(self.beam_width, max_nodes)
                )

                for log_prob, node_idx in zip(topk_log_probs, topk_indices):
                    node_idx = node_idx.item()

                    # Skip if already visited
                    if visited_mask[0, node_idx] == 0:
                        continue

                    # Create new beam
                    new_sequence = sequence + [node_idx]
                    new_score = score + log_prob.item()

                    # Update visited mask
                    new_visited_mask = visited_mask.clone()
                    new_visited_mask[0, node_idx] = 0

                    # Get next decoder input
                    next_node_emb = node_embeddings_padded[0, node_idx].unsqueeze(0).unsqueeze(0)

                    candidates.append((
                        new_sequence,
                        new_score,
                        new_lstm_state,
                        new_visited_mask,
                        next_node_emb
                    ))

            # Keep top beams
            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]

        # Return best sequence
        if beams:
            best_sequence, best_score, _, _, _ = beams[0]
            return best_sequence, best_score
        else:
            return [], float('-inf')
