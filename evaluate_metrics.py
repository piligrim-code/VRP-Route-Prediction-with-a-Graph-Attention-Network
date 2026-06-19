"""
Evaluate a trained VRP model on the validation split using sequence-level metrics.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_loader import VRPDataLoader
from data.vrp_node_dataset import VRPNodeDataset
from models.attention_vrp import AttentionVRP
from utils.advanced_metrics import compute_all_metrics, print_metrics

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
if not Path(checkpoint_path).exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from {checkpoint_path}")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val accuracy: {checkpoint['metrics']['accuracy']:.4f}\n")

print("Loading validation data...")
data_loader = VRPDataLoader('dataset')
all_pairs = data_loader.load_scenario_pairs()
split_idx = int(len(all_pairs) * 0.8)
val_pairs = all_pairs[split_idx:]
val_dataset = VRPNodeDataset(val_pairs)
print(f"Validation samples: {len(val_dataset)}\n")

# Run inference one sample at a time; predicted and target sequence lengths
# differ per scenario, so pad each pair to a common length before stacking.
print("Running inference...")
all_predictions = []
all_targets = []

with torch.no_grad():
    for data in val_dataset:
        data = data.to(device)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

        _, predictions, _ = model(
            data.x, data.edge_index, data.edge_attr, batch,
            target_nodes=None,
            teacher_forcing_ratio=0.0
        )

        pred_seq = predictions[0]
        target_seq = data.y
        max_len = max(pred_seq.shape[0], target_seq.shape[0])

        pred_padded = torch.full((max_len,), -1, device=device)
        pred_padded[:pred_seq.shape[0]] = pred_seq[:max_len]
        target_padded = torch.full((max_len,), -1, device=device)
        target_padded[:target_seq.shape[0]] = target_seq[:max_len]

        all_predictions.append(pred_padded)
        all_targets.append(target_padded)

# Pad all sequences to the global maximum length before stacking.
max_seq_len = max(p.shape[0] for p in all_predictions)
predictions = torch.full((len(all_predictions), max_seq_len), -1, device=device)
targets = torch.full((len(all_targets), max_seq_len), -1, device=device)
for i, (p, t) in enumerate(zip(all_predictions, all_targets)):
    predictions[i, :p.shape[0]] = p
    targets[i, :t.shape[0]] = t

print(f"Total predictions: {predictions.shape[0]} sequences\n")

metrics = compute_all_metrics(predictions, targets, ignore_index=-1)
print_metrics(metrics, prefix="Validation")
