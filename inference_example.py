"""
Inference example: run a trained model on one scenario and compare the
predicted node sequence against the ground-truth route.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_loader import VRPDataLoader
from data.vrp_node_dataset import VRPNodeDataset
from models.attention_vrp import AttentionVRP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

model = AttentionVRP(
    node_features=12,
    edge_features=3,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.25,
    max_route_length=50
).to(device)

checkpoint_path = 'checkpoints/attention_vrp_best.pt'
if Path(checkpoint_path).exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val accuracy: {checkpoint['metrics']['accuracy']:.4f}\n")
else:
    print(f"Checkpoint not found: {checkpoint_path}")
    print("Train a checkpoint with saved scaler_state before inference.\n")
    sys.exit(1)

model.eval()

print("Loading example scenario...")
data_loader = VRPDataLoader('dataset')
scenario_pairs = data_loader.load_scenario_pairs(max_samples=1)

if len(scenario_pairs) == 0:
    print("No scenarios found.")
    sys.exit(1)

input_scenario, output_scenario = scenario_pairs[0]
print(f"Loaded scenario: {input_scenario.get('scenario_id', 'unknown')}\n")

dataset = VRPNodeDataset(scenario_pairs, fit_scalers=False, scaler_state=checkpoint.get('scaler_state'))
if len(dataset) == 0:
    print("Failed to create dataset.")
    sys.exit(1)

graph_data = dataset[0]
nodes = input_scenario['nodes']
edges = input_scenario['edges']

print("Input graph:")
print(f"  Nodes: {graph_data.x.shape[0]}")
print(f"  Edges: {graph_data.edge_index.shape[1]}")
print(f"  Node feature dim: {graph_data.x.shape[1]}")
print(f"  Edge feature dim: {graph_data.edge_attr.shape[1]}\n")

# Run inference
with torch.no_grad():
    graph_data = graph_data.to(device)
    batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=device)
    _, predictions, _ = model(
        graph_data.x, graph_data.edge_index, graph_data.edge_attr, batch,
        target_nodes=None,
        teacher_forcing_ratio=0.0
    )
    predicted_nodes = predictions[0].cpu().numpy()

# Map node indices to ids
idx_to_node_id = {}
node_id_to_idx = {}
for idx, node in enumerate(nodes):
    node_id = node.get('node_id', f'node_{idx}')
    node_id_to_idx[node_id] = idx
    idx_to_node_id[idx] = node_id

print(f"Predicted route length: {len(predicted_nodes)} nodes")
print("Predicted node sequence (first 15):")
for i, node_idx in enumerate(predicted_nodes[:15]):
    node_id = idx_to_node_id.get(int(node_idx), f'node_{node_idx}')
    node_info = nodes[int(node_idx)] if int(node_idx) < len(nodes) else {}
    node_type = node_info.get('type', 'unknown')
    print(f"  {i + 1}. {node_id} ({node_type})")
if len(predicted_nodes) > 15:
    print(f"  ... and {len(predicted_nodes) - 15} more nodes")
print()

# Build ground-truth node sequence from the route edges
route_edges = output_scenario.get('route', [])
gt_sequence = []
for edge_idx in route_edges:
    if edge_idx >= len(edges):
        continue
    edge = edges[edge_idx]
    src_id = edge.get('from_node_id', '')
    dst_id = edge.get('to_node_id', '')
    if len(gt_sequence) == 0:
        gt_sequence.append(node_id_to_idx.get(src_id, 0))
    gt_sequence.append(node_id_to_idx.get(dst_id, 0))

print(f"Ground-truth route length: {len(gt_sequence)} nodes\n")

# Position-level accuracy on the overlapping prefix
min_len = min(len(predicted_nodes), len(gt_sequence))
correct = sum(1 for i in range(min_len) if predicted_nodes[i] == gt_sequence[i])
accuracy = correct / min_len if min_len > 0 else 0.0

print("Comparison:")
print(f"  Matching positions: {correct}/{min_len}")
print(f"  Position accuracy: {accuracy * 100:.2f}%")
