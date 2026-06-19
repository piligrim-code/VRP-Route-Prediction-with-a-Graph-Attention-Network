"""
Training script for the attention VRP model with a route-cost loss term.

Trains AttentionVRP on node-level route prediction using cross-entropy plus
an auxiliary route-cost penalty. Saves the best checkpoint by validation accuracy.
"""
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_loader import VRPDataLoader
from data.vrp_node_dataset import VRPNodeDataset
from models.attention_vrp import AttentionVRP
from training.route_cost_loss import RouteCostLoss
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 30,
    'alpha': 0.1,  # route-cost weight in the loss
    'patience': 5,
}

print(f"Device: {config['device']}")
print(f"Route-cost weight (alpha): {config['alpha']}\n")

print("Loading data...")
data_loader = VRPDataLoader('dataset')
all_pairs = data_loader.load_scenario_pairs()

split_idx = int(len(all_pairs) * 0.8)
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

print(f"Train samples: {len(train_pairs)}")
print(f"Val samples: {len(val_pairs)}\n")

train_dataset = VRPNodeDataset(train_pairs, fit_scalers=True)
val_dataset = VRPNodeDataset(val_pairs, fit_scalers=True)


def collate_fn(batch_list):
    """Keep variable-length targets separate from the batched graph."""
    from torch_geometric.data import Batch

    targets = [data.y.clone() for data in batch_list]
    batch = Batch.from_data_list(batch_list)
    return batch, targets


train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}\n")

print("Creating model...")
model = AttentionVRP(
    node_features=12,  # 5 base + 7 order features
    edge_features=3,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.25,
    max_route_length=50
).to(config['device'])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

criterion = RouteCostLoss(alpha=config['alpha'])
optimizer = Adam(model.parameters(), lr=config['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

print("Starting training...\n")

best_val_acc = 0.0
patience_counter = 0

for epoch in range(config['num_epochs']):
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")

    # Training
    model.train()
    train_loss = 0.0
    train_ce_loss = 0.0
    train_route_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch, targets_list in tqdm(train_loader, desc="Training"):
        batch_size = len(targets_list)
        batch = batch.to(config['device'])
        optimizer.zero_grad()

        max_len = max(len(t) for t in targets_list)
        targets_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=config['device'])
        for i, target in enumerate(targets_list):
            targets_padded[i, :len(target)] = target.to(config['device'])

        logits, predictions, _ = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            target_nodes=targets_padded,
            teacher_forcing_ratio=0.5
        )

        loss, metrics = criterion(
            logits, predictions, targets_padded,
            batch.edge_attr, batch.edge_index, batch.batch
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += metrics['total_loss']
        train_ce_loss += metrics['ce_loss']
        train_route_loss += metrics['route_cost_loss']

        for i in range(len(predictions)):
            pred_seq = predictions[i]
            target_seq = targets_list[i].to(config['device'])
            min_len = min(len(pred_seq), len(target_seq))
            train_correct += sum(1 for j in range(min_len) if pred_seq[j] == target_seq[j])
            train_total += len(target_seq)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_ce = train_ce_loss / len(train_loader)
    avg_train_route = train_route_loss / len(train_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0

    print(f"Train Loss: {avg_train_loss:.4f} (CE: {avg_train_ce:.4f}, Route: {avg_train_route:.4f})")
    print(f"Train Accuracy: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_ce_loss = 0.0
    val_route_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch, targets_list in tqdm(val_loader, desc="Validation"):
            batch_size = len(targets_list)
            batch = batch.to(config['device'])

            max_len = max(len(t) for t in targets_list)
            targets_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=config['device'])
            for i, target in enumerate(targets_list):
                targets_padded[i, :len(target)] = target.to(config['device'])

            logits, predictions, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                target_nodes=None,
                teacher_forcing_ratio=0.0
            )

            loss, metrics = criterion(
                logits, predictions, targets_padded,
                batch.edge_attr, batch.edge_index, batch.batch
            )

            val_loss += metrics['total_loss']
            val_ce_loss += metrics['ce_loss']
            val_route_loss += metrics['route_cost_loss']

            for i in range(len(predictions)):
                pred_seq = predictions[i]
                target_seq = targets_list[i].to(config['device'])
                min_len = min(len(pred_seq), len(target_seq))
                val_correct += sum(1 for j in range(min_len) if pred_seq[j] == target_seq[j])
                val_total += len(target_seq)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_ce = val_ce_loss / len(val_loader)
    avg_val_route = val_route_loss / len(val_loader)
    val_acc = val_correct / val_total if val_total > 0 else 0

    print(f"Val Loss: {avg_val_loss:.4f} (CE: {avg_val_ce:.4f}, Route: {avg_val_route:.4f})")
    print(f"Val Accuracy: {val_acc:.4f}\n")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        Path('checkpoints').mkdir(exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': {
                'accuracy': val_acc,
                'loss': avg_val_loss,
                'ce_loss': avg_val_ce,
                'route_loss': avg_val_route,
            },
        }
        torch.save(checkpoint, 'checkpoints/attention_vrp_best.pt')
        print(f"Saved best model (val_acc={val_acc:.4f})\n")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{config['patience']})\n")

    if patience_counter >= config['patience']:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
print("Model saved to: checkpoints/attention_vrp_best.pt")
