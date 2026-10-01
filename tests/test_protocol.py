import json

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from data.vrp_node_dataset import VRPNodeDataset
from models.attention_vrp import AttentionDecoder, AttentionVRP
from training.route_cost_loss import RouteCostLoss


def scenario(n=2, shift=0):
    nodes = [{'node_id': str(i), 'coords': [shift + i, i * 2], 'type': 'base' if i == 0 else 'pickup'} for i in range(n)]
    edges = [{'edge_id': f'{i}-{j}', 'from_node_id': str(i), 'to_node_id': str(j),
              'distance_km': float(i + j + 1), 'duration_minutes': 2.0, 'toll_cost_rub': 0.0}
             for i in range(n) for j in range(n) if i != j]
    route = [next(k for k, edge in enumerate(edges) if edge['from_node_id'] == str(i) and edge['to_node_id'] == str(i + 1)) for i in range(n - 1)]
    return {'nodes': nodes, 'edges': edges, 'orders': []}, {'route': route}


def test_validation_uses_training_statistics_without_refit():
    train = VRPNodeDataset([scenario()])
    state = train.scaler_state()
    validation = VRPNodeDataset([scenario(shift=100)], fit_scalers=False, scaler_state=state)
    assert train.scaler_state() == validation.scaler_state() == state
    assert validation[0].x[:, 0].mean() > 100
    assert torch.equal(validation[0].raw_edge_costs, torch.tensor([2., 2.]))
    assert torch.equal(validation[0].edge_attr[:, 0], torch.zeros(2))


def test_scaler_checkpoint_roundtrip_is_weights_only_safe(tmp_path):
    train = VRPNodeDataset([scenario(3)])
    state = json.loads(json.dumps(train.scaler_state()))
    file = tmp_path / 'fixture.pt'
    torch.save({'scaler_state': state}, file)
    loaded = torch.load(file, weights_only=True)
    clone = VRPNodeDataset([scenario(3)], fit_scalers=False, scaler_state=loaded['scaler_state'])
    assert torch.equal(train[0].x, clone[0].x)
    assert torch.equal(train[0].edge_attr, clone[0].edge_attr)


def test_inference_needs_no_ground_truth():
    train = VRPNodeDataset([scenario()])
    data, _ = scenario()
    inference = VRPNodeDataset([(data, {})], fit_scalers=False, scaler_state=train.scaler_state(), require_targets=False)
    assert len(inference) == 1 and inference[0].y.numel() == 0


def test_missing_scaler_cannot_silently_fit_validation():
    with pytest.raises(ValueError, match='scaler_state'):
        VRPNodeDataset([scenario()], fit_scalers=False)


def test_supplied_scaler_cannot_be_refit():
    state = VRPNodeDataset([scenario()]).scaler_state()
    with pytest.raises(ValueError, match='refit'):
        VRPNodeDataset([scenario()], scaler_state=state)


def test_invalid_scaler_statistics():
    state = VRPNodeDataset([scenario()]).scaler_state()
    state['node']['scale_'][0] = 0
    with pytest.raises(ValueError, match='statistics'):
        VRPNodeDataset([scenario()], fit_scalers=False, scaler_state=state)


def loss_inputs():
    logits = torch.tensor([[[0.2, 1.0], [1.3, -0.2]]], requires_grad=True)
    targets = torch.tensor([[0, 1]])
    edges = torch.tensor([[0, 1], [1, 0]])
    batch = torch.tensor([0, 0])
    costs = torch.tensor([1., 4.])
    return logits, targets, edges, batch, costs


def test_route_surrogate_contributes_nonzero_gradient():
    logits, targets, edges, batch, costs = loss_inputs()
    total, _ = RouteCostLoss(0.1)(logits, logits.argmax(-1), targets, None, edges, batch, costs)
    grad = torch.autograd.grad(total, logits, retain_graph=True)[0]
    ce = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    grad_ce = torch.autograd.grad(ce, logits)[0]
    assert torch.isfinite(grad).all()
    assert (grad - 0.9 * grad_ce).abs().max() > 1e-4


def test_graph_local_costs_in_a_batch():
    logits, targets, edges, batch, costs = loss_inputs()
    loss = RouteCostLoss(1.0)
    first, _ = loss(logits, None, targets, None, edges, batch, costs)
    second, _ = loss(logits.flip(-1), None, targets, None, edges, batch, costs * 3)
    combined, _ = loss(torch.cat([logits, logits.flip(-1)]), None, targets.repeat(2, 1), None,
                       torch.cat([edges, edges + 2], dim=1), torch.tensor([0, 0, 1, 1]),
                       torch.cat([costs, costs * 3]))
    assert torch.allclose(combined, (first + second) / 2)


def test_normalized_edge_features_cannot_replace_raw_costs():
    logits, targets, edges, batch, _ = loss_inputs()
    with pytest.raises(ValueError, match='raw_edge_costs'):
        RouteCostLoss()(logits, None, targets, torch.zeros(2, 3), edges, batch)


def test_padding_is_finite_and_ignored():
    logits, targets, edges, batch, costs = loss_inputs()
    padded = torch.cat([logits, torch.full((1, 1, 2), -torch.inf)], dim=1)
    a, _ = RouteCostLoss()(logits, None, targets, None, edges, batch, costs)
    b, _ = RouteCostLoss()(padded, None, torch.tensor([[0, 1, -1]]), None, edges, batch, costs)
    assert torch.allclose(a, b)
    b.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_labels_have_zero_finite_gradient():
    logits, targets, edges, batch, costs = loss_inputs()
    loss, _ = RouteCostLoss()(logits, None, torch.full_like(targets, -1), None, edges, batch, costs)
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


def test_cross_graph_edges_rejected():
    logits, targets, edges, _, costs = loss_inputs()
    with pytest.raises(ValueError, match='cross graphs'):
        RouteCostLoss()(logits.repeat(2, 1, 1), None, targets.repeat(2, 1), None,
                        edges, torch.tensor([0, 1]), costs)


def test_attention_heads_keep_their_own_graph_mask():
    torch.manual_seed(1)
    decoder = AttentionDecoder(8, num_heads=2).eval()
    states, nodes = torch.randn(2, 8), torch.randn(2, 3, 8)
    masks = torch.tensor([[1., 1., 0.], [0., 1., 1.]])
    combined, _ = decoder(states, nodes, masks)
    singles = torch.cat([decoder(states[i:i+1], nodes[i:i+1], masks[i:i+1])[0] for i in range(2)])
    assert torch.allclose(combined, singles, atol=1e-6)


def test_variable_graphs_forward_backward_on_cpu():
    torch.manual_seed(7)
    dataset = VRPNodeDataset([scenario(2), scenario(3)])
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    target = torch.tensor([[0, 1, -1], [0, 1, 2]])
    model = AttentionVRP(node_features=12, edge_features=3, hidden_dim=16, num_heads=2,
                         num_layers=1, dropout=0, max_route_length=4)
    logits, prediction, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                  target_nodes=target, teacher_forcing_ratio=1.0)
    loss, _ = RouteCostLoss()(logits, prediction, target, batch.edge_attr, batch.edge_index,
                              batch.batch, batch.raw_edge_costs)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        out, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                          teacher_forcing_ratio=0, decode_steps=3)
    assert out.shape == (2, 3, 3)


def test_pipeline_reuses_checkpoint_scalers_without_labels(tmp_path):
    from inference_pipeline import VRPInferencePipeline

    torch.manual_seed(11)
    train = VRPNodeDataset([scenario(3)])
    model = AttentionVRP(node_features=12, edge_features=3, hidden_dim=256,
                         num_heads=8, num_layers=4, dropout=0.25, max_route_length=50)
    path = tmp_path / 'synthetic_untrained.pt'
    torch.save({'model_state_dict': model.state_dict(), 'metrics': {'accuracy': 0.0},
                'scaler_state': train.scaler_state()}, path)
    pipeline = VRPInferencePipeline(str(path), device='cpu')
    before = json.dumps(pipeline.scalers, sort_keys=True)
    first = pipeline.predict_single(scenario(3)[0], return_confidence=False)
    second = pipeline.predict_single(scenario(3, shift=10)[0], return_confidence=False)
    assert len(first['predicted_route']) == len(second['predicted_route']) == 3
    assert all(0 <= i < 3 for i in first['predicted_route'])
    assert json.dumps(pipeline.scalers, sort_keys=True) == before


def test_legacy_checkpoint_does_not_fit_scalers_on_inference(tmp_path):
    from inference_pipeline import VRPInferencePipeline

    path = tmp_path / 'legacy.pt'
    torch.save({'metrics': {'accuracy': 0.0}}, path)
    with pytest.raises(ValueError, match='scaler_state'):
        VRPInferencePipeline(str(path), device='cpu')
