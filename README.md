# VRP Route Prediction with a Graph Attention Network

A learned solver for the Vehicle Routing Problem (VRP). Given a graph of
locations (bases, pickups, deliveries) and the edges between them, the model
predicts the sequence of nodes to visit, for example
`base -> pickup_1 -> delivery_1 -> base`.

The task is framed as node-level autoregressive prediction rather than
edge-level prediction. Predicting nodes keeps the output vocabulary small
(the nodes in a single graph, typically 3-30) instead of the hundreds of
candidate edges, which made the problem tractable. A visited mask enforces the
VRP constraint that each node is selected at most once.

## Architecture

- **Encoder:** 4 layers of GATv2 graph attention (8 heads, hidden dimension
  256) with residual connections and layer normalization. Edge features are
  used inside the attention computation.
- **Decoder:** an LSTM maintains the decoder state; a multi-head attention
  module scores the remaining nodes at each step and selects the next one.
  A visited mask prevents revisiting nodes. Teacher forcing is used during
  training.
- About 1.8M parameters.

## Result

The best checkpoint reaches roughly 78% position-level accuracy on the
validation split. Sequence-level (exact full-route match) accuracy is lower;
`evaluate_metrics.py` reports both along with edit distance, LCS ratio, and
related metrics.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.0+, and PyTorch Geometric 2.3+. A GPU is
optional; the scripts fall back to CPU.

## Dataset format

Place the data under `dataset/` with input scenarios and their results in
paired folders (the loader matches `input <range>` with `output <range>` or a
plain `<range>` folder containing `RESULT_*.json` files).

Input scenario (`scenario_XXXX.json`):

```json
{
  "scenario_id": "scenario_0001",
  "nodes": [
    {"node_id": "base_1", "type": "base", "coords": [55.75, 37.62], "service_time_min": 0},
    {"node_id": "pickup_1", "type": "pickup", "coords": [55.82, 36.87], "max_capacity": 10}
  ],
  "edges": [
    {"edge_id": "edge_1", "from_node_id": "base_1", "to_node_id": "pickup_1",
     "distance_km": 45.2, "duration_minutes": 67.3, "toll_cost_rub": 150.0}
  ],
  "orders": []
}
```

Output result (`RESULT_scenario_XXXX.json`):

```json
{
  "vehicleTypeToOptimalRouteList": {
    "car_1": [{"route": ["edge_1", "edge_2", "edge_3"]}]
  }
}
```

The data loader converts the nested edge-id route into a list of node indices
used as the training target.

### Features

Node features (12): x, y coordinates, max capacity, is-base flag, service time,
and 7 order-derived flags (is-delivery, is-pickup, is-solid, is-consequence,
is-divisible, is-multipickup, total quantity). Edge features (3): distance,
duration, toll cost. All features are standardized with a `StandardScaler`.

## Usage

Train (writes the best checkpoint to `checkpoints/attention_vrp_best.pt`):

```bash
python train_with_cost.py
```

Run inference on a single scenario and compare against ground truth:

```bash
python inference_example.py
```

Evaluate on the validation split:

```bash
python evaluate_metrics.py
```

Metrics reported by the evaluator:

- Position accuracy: fraction of correctly predicted nodes per position.
- Sequence accuracy: fraction of fully correct routes.
- Prefix-k accuracy: accuracy over the first k nodes.
- Normalized edit distance, LCS ratio, first-error position, node coverage.

## Project layout

```
optimize_logic/
├── src/
│   ├── data/
│   │   ├── data_loader.py        # load and pair input/output JSON scenarios
│   │   └── vrp_node_dataset.py   # build graphs, node sequence targets, scaling
│   ├── models/
│   │   └── attention_vrp.py      # GAT encoder + attention/LSTM decoder
│   ├── training/
│   │   └── route_cost_loss.py    # cross-entropy + route-cost loss
│   ├── inference/
│   │   └── beam_search.py        # optional beam-search decoder
│   ├── utils/
│   │   └── advanced_metrics.py   # sequence-level metrics
│   └── inference_pipeline.py     # reusable inference wrapper
├── train_with_cost.py            # training entry point
├── inference_example.py          # single-scenario inference demo
├── evaluate_metrics.py           # validation-set evaluation
├── config.yaml                   # configuration reference
└── requirements.txt
```

## License

MIT. See [LICENSE](LICENSE).
