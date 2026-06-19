"""
VRP Dataset for node-level prediction.
Converts edge sequences to node sequences.
"""
import torch
from torch_geometric.data import Data
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VRPNodeDataset:
    """
    Dataset that converts edge-based routes to node-based routes.
    """

    def __init__(self, scenario_pairs, fit_scalers=True):
        """
        Args:
            scenario_pairs: List of (input_scenario, output_scenario) tuples
            fit_scalers: If True, fit scalers on this data. Set False for test set.
        """
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.data = []

        # First pass: collect all features to fit scalers
        if fit_scalers:
            all_node_features = []
            all_edge_features = []

            for input_scenario, _ in scenario_pairs:
                nodes = input_scenario.get('nodes', [])
                edges = input_scenario.get('edges', [])

                # Build node_id to index mapping for order extraction
                node_id_to_idx = {}
                for idx, node in enumerate(nodes):
                    node_id = node.get('node_id', f'node_{idx}')
                    node_id_to_idx[node_id] = idx

                # Extract order features
                order_features = self._extract_order_features(input_scenario, node_id_to_idx)

                for idx, node in enumerate(nodes):
                    coords = node.get('coords', [0.0, 0.0])
                    base_features = [
                        coords[0] if len(coords) > 0 else 0.0,
                        coords[1] if len(coords) > 1 else 0.0,
                        node.get('max_capacity', 0.0),
                        1.0 if node.get('type', '') == 'base' else 0.0,
                        node.get('service_time_min', 0.0)
                    ]
                    # Combine base features with order features
                    features = base_features + order_features[idx]
                    all_node_features.append(features)

                for edge in edges:
                    features = [
                        edge.get('distance_km', 0.0),
                        edge.get('duration_minutes', 0.0),
                        edge.get('toll_cost_rub', 0.0)
                    ]
                    all_edge_features.append(features)

            # Fit scalers
            if len(all_node_features) > 0:
                self.node_scaler.fit(all_node_features)
            if len(all_edge_features) > 0:
                self.edge_scaler.fit(all_edge_features)

            logger.info(f"Fitted scalers on {len(all_node_features)} nodes, {len(all_edge_features)} edges")

        # Second pass: create graphs with normalized features
        for input_scenario, output_scenario in scenario_pairs:
            graph_data = self._create_graph(input_scenario, output_scenario)
            if graph_data is not None:
                self.data.append(graph_data)

        logger.info(f"Initialized VRPNodeDataset with {len(self.data)} scenarios")

    def _extract_order_features(self, input_scenario, node_id_to_idx):
        """Extract order-related features for each node."""
        orders = input_scenario.get('orders', [])
        num_nodes = len(node_id_to_idx)

        # Initialize features for each node
        # [is_delivery, is_pickup, is_solid, is_consequence, is_divisible, is_multipickup, total_quantity]
        order_features = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(num_nodes)]

        for order in orders:
            is_consequence = 1.0 if order.get('is_consequence', False) else 0.0

            for node_option in order.get('node_options', []):
                node_id = node_option.get('node_id', '')
                node_idx = node_id_to_idx.get(node_id)

                if node_idx is None:
                    continue

                # Mark as delivery node
                order_features[node_idx][0] = 1.0

                # is_solid flag
                is_solid = 1.0 if node_option.get('is_solid', False) else 0.0
                order_features[node_idx][2] = max(order_features[node_idx][2], is_solid)

                # is_consequence flag
                order_features[node_idx][3] = max(order_features[node_idx][3], is_consequence)

                # Process items
                total_quantity = 0.0
                for item_option in node_option.get('item_options', []):
                    quantity = item_option.get('quantity', 0)
                    total_quantity += quantity

                    is_divisible = 1.0 if item_option.get('is_divisible', False) else 0.0
                    is_multipickup = 1.0 if item_option.get('is_multipickup', False) else 0.0

                    order_features[node_idx][4] = max(order_features[node_idx][4], is_divisible)
                    order_features[node_idx][5] = max(order_features[node_idx][5], is_multipickup)

                    # Mark pickup nodes
                    for pickup_option in item_option.get('pickup_options', []):
                        pickup_node_id = pickup_option.get('node_id', '')
                        pickup_idx = node_id_to_idx.get(pickup_node_id)
                        if pickup_idx is not None:
                            order_features[pickup_idx][1] = 1.0  # is_pickup

                # Store total quantity
                order_features[node_idx][6] = total_quantity

        return order_features

    def _create_graph(self, input_scenario, output_scenario):
        """Create PyG Data object with node sequence as target."""
        try:
            # Extract nodes and edges
            nodes = input_scenario['nodes']
            edges = input_scenario['edges']

            if len(nodes) == 0 or len(edges) == 0:
                return None

            # Build node_id to index mapping
            node_id_to_idx = {}
            for idx, node in enumerate(nodes):
                node_id = node.get('node_id', f'node_{idx}')
                node_id_to_idx[node_id] = idx

            # Extract order features
            order_features = self._extract_order_features(input_scenario, node_id_to_idx)

            # Node features: [x, y, max_capacity, is_base, service_time, is_delivery, is_pickup, is_solid, is_consequence, is_divisible, is_multipickup, total_quantity]
            node_features = []
            for idx, node in enumerate(nodes):
                coords = node.get('coords', [0.0, 0.0])
                base_features = [
                    coords[0] if len(coords) > 0 else 0.0,
                    coords[1] if len(coords) > 1 else 0.0,
                    node.get('max_capacity', 0.0),
                    1.0 if node.get('type', '') == 'base' else 0.0,
                    node.get('service_time_min', 0.0)
                ]
                # Combine base features with order features
                features = base_features + order_features[idx]
                node_features.append(features)

            # Normalize node features
            node_features = self.node_scaler.transform(node_features)
            x = torch.tensor(node_features, dtype=torch.float32)

            # Edge index and features
            edge_index = []
            edge_features = []

            for edge in edges:
                src_id = edge.get('from_node_id', '')
                dst_id = edge.get('to_node_id', '')

                src = node_id_to_idx.get(src_id, 0)
                dst = node_id_to_idx.get(dst_id, 0)

                if src >= len(nodes) or dst >= len(nodes):
                    continue

                edge_index.append([src, dst])

                features = [
                    edge.get('distance_km', 0.0),
                    edge.get('duration_minutes', 0.0),
                    edge.get('toll_cost_rub', 0.0)
                ]
                edge_features.append(features)

            if len(edge_index) == 0:
                return None

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Normalize edge features
            edge_features = self.edge_scaler.transform(edge_features)
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)

            # Convert edge route to node sequence
            route_edges = output_scenario.get('route', [])
            node_sequence = self._edges_to_nodes(route_edges, edges, node_id_to_idx)

            if len(node_sequence) == 0:
                return None

            y = torch.tensor(node_sequence, dtype=torch.long)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        except Exception as e:
            logger.warning(f"Failed to create graph: {e}")
            return None

    def _edges_to_nodes(self, route_edges, all_edges, node_id_to_idx):
        """
        Convert edge indices to node sequence.

        Args:
            route_edges: List of edge indices in the route
            all_edges: List of all edge dictionaries
            node_id_to_idx: Mapping from node_id to index

        Returns:
            List of node indices forming the route
        """
        if len(route_edges) == 0:
            logger.debug("Empty route_edges")
            return []

        node_sequence = []
        skipped = 0

        for edge_idx in route_edges:
            if edge_idx >= len(all_edges):
                skipped += 1
                continue

            edge = all_edges[edge_idx]
            src_id = edge.get('from_node_id', '')
            dst_id = edge.get('to_node_id', '')

            src = node_id_to_idx.get(src_id, 0)
            dst = node_id_to_idx.get(dst_id, 0)

            # Add source node if it's the first edge
            if len(node_sequence) == 0:
                node_sequence.append(src)

            # Add destination node
            node_sequence.append(dst)

        if len(node_sequence) == 0 and len(route_edges) > 0:
            logger.warning(f"Failed to convert edges to nodes: {len(route_edges)} edges, {skipped} skipped")

        return node_sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

