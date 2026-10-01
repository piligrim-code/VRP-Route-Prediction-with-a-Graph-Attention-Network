"""
Attention-based model for VRP (node-level prediction).
Based on "Attention, Learn to Solve Routing Problems" paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
import logging

logger = logging.getLogger(__name__)


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder that selects next node to visit.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Query projection (from decoder state)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Key/Value projections (from node embeddings)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_state, node_embeddings, mask=None):
        """
        Args:
            decoder_state: [batch_size, hidden_dim] - current state
            node_embeddings: [batch_size, num_nodes, hidden_dim]
            mask: [batch_size, num_nodes] - 1 for valid nodes, 0 for visited

        Returns:
            logits: [batch_size, num_nodes] - scores for each node
            attention_weights: [batch_size, num_nodes]
        """
        batch_size, num_nodes, _ = node_embeddings.shape

        # Query from decoder state
        query = self.query_proj(decoder_state).unsqueeze(1)  # [B, 1, H]

        # Keys and values from node embeddings
        keys = self.key_proj(node_embeddings)  # [B, N, H]
        values = self.value_proj(node_embeddings)  # [B, N, H]

        # Attention mask: convert to additive mask
        if mask is not None:
            # mask: 1 for valid, 0 for invalid
            # attention needs: 0 for valid, -inf for invalid
            attn_mask = (1 - mask) * -1e9
            # Expand for multi-head: [B, 1, N] -> [B*num_heads, 1, N]
            attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, dim=0)  # [B*H, 1, N]
        else:
            attn_mask = None

        # Multi-head attention
        attended, attn_weights = self.attention(
            query, keys, values,
            key_padding_mask=None,
            attn_mask=attn_mask
        )

        # Output projection
        output = self.out_proj(attended.squeeze(1))  # [B, H]

        # Compute logits as compatibility scores
        logits = torch.bmm(
            output.unsqueeze(1),  # [B, 1, H]
            node_embeddings.transpose(1, 2)  # [B, H, N]
        ).squeeze(1)  # [B, N]

        # Apply mask to logits
        if mask is not None:
            logits = logits + (1 - mask) * -1e9

        return logits, attn_weights.squeeze(1)


class AttentionVRP(nn.Module):
    """
    Attention-based VRP model with node-level prediction.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.25,
        max_route_length: int = 50
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_route_length = max_route_length

        # Node feature projection
        self.node_proj = nn.Linear(node_features, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        # GAT encoder layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Attention decoder
        self.decoder = AttentionDecoder(hidden_dim, num_heads)

        # LSTM for decoder state
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        logger.info(f"Initialized AttentionVRP: {num_layers} layers, {num_heads} heads, hidden_dim={hidden_dim}")

    def encode_graph(self, x, edge_index, edge_attr, batch):
        """
        Encode graph with GAT layers.

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        # Project features
        node_emb = self.node_proj(x)
        edge_emb = self.edge_proj(edge_attr)

        # Apply GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            node_emb_new = gat(node_emb, edge_index, edge_attr=edge_emb)
            node_emb_new = norm(node_emb_new)
            node_emb = node_emb + node_emb_new if i > 0 else node_emb_new
            node_emb = F.relu(node_emb)

        return node_emb

    def forward(self, x, edge_index, edge_attr, batch, target_nodes=None, teacher_forcing_ratio=0.8, decode_steps=None):
        """
        Args:
            x: [num_nodes, node_features]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_features]
            batch: [num_nodes] - batch assignment
            target_nodes: [batch_size, seq_len] - target node sequence (optional)
            teacher_forcing_ratio: float

        Returns:
            logits: [batch_size, seq_len, num_nodes_per_graph]
            predictions: [batch_size, seq_len]
            attention_weights: [batch_size, seq_len, num_nodes_per_graph]
        """
        device = x.device

        # Encode graph
        node_embeddings = self.encode_graph(x, edge_index, edge_attr, batch)

        # Get batch info
        num_graphs = batch.max().item() + 1
        batch_size = num_graphs

        # Organize node embeddings by graph
        node_embeddings_batched = []
        num_nodes_per_graph = []
        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = node_embeddings[mask]
            node_embeddings_batched.append(graph_nodes)
            num_nodes_per_graph.append(graph_nodes.size(0))

        # Pad to same length
        max_nodes = max(num_nodes_per_graph)
        node_embeddings_padded = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        for i, emb in enumerate(node_embeddings_batched):
            node_embeddings_padded[i, :emb.size(0)] = emb

        # Determine sequence length
        if target_nodes is not None:
            max_seq_len = target_nodes.size(1)
        elif decode_steps is not None:
            if isinstance(decode_steps, bool) or not isinstance(decode_steps, int) or decode_steps <= 0:
                raise ValueError('decode_steps must be a positive integer')
            max_seq_len = decode_steps
        else:
            max_seq_len = min(self.max_route_length, max_nodes)

        # Initialize decoder state
        decoder_input = self.start_token.expand(batch_size, 1, -1)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        lstm_state = (h_0, c_0)

        # Initialize visited mask (1 = can visit, 0 = visited)
        visited_mask = torch.ones(batch_size, max_nodes, device=device)
        for i, n in enumerate(num_nodes_per_graph):
            visited_mask[i, n:] = 0  # Mask padding nodes

        all_logits = []
        all_predictions = []
        all_attention = []

        # Autoregressive decoding
        for t in range(max_seq_len):
            # LSTM step
            lstm_out, lstm_state = self.lstm(decoder_input, lstm_state)
            decoder_state = lstm_out.squeeze(1)  # [batch_size, hidden_dim]

            # Attention to select next node
            logits, attn_weights = self.decoder(
                decoder_state,
                node_embeddings_padded,
                mask=visited_mask
            )

            all_logits.append(logits)
            all_attention.append(attn_weights)

            # Sample or use target
            if target_nodes is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing
                next_node = target_nodes[:, t]
            else:
                # Greedy sampling
                next_node = torch.argmax(logits, dim=-1)

            # Clamp to valid range
            next_node = torch.clamp(next_node, 0, max_nodes - 1)

            all_predictions.append(next_node)

            # Update visited mask
            visited_mask.scatter_(1, next_node.unsqueeze(1), 0)

            # Get embedding of selected node for next step
            next_node_emb = node_embeddings_padded[torch.arange(batch_size, device=device), next_node]
            decoder_input = next_node_emb.unsqueeze(1)

        # Stack outputs
        logits = torch.stack(all_logits, dim=1)  # [batch_size, seq_len, num_nodes]
        predictions = torch.stack(all_predictions, dim=1)  # [batch_size, seq_len]
        attention_weights = torch.stack(all_attention, dim=1)  # [batch_size, seq_len, num_nodes]

        return logits, predictions, attention_weights


