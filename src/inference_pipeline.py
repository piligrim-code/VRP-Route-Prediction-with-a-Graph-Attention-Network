"""
VRP inference pipeline for running a trained model on new scenarios.
"""
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from data.data_loader import VRPDataLoader
from data.vrp_node_dataset import VRPNodeDataset
from models.attention_vrp import AttentionVRP
from inference.beam_search import BeamSearchVRP

logger = logging.getLogger(__name__)


class VRPInferencePipeline:
    """
    Pipeline for VRP inference.

    Loads the model once, caches feature scalers, and supports batch
    processing on CPU or GPU.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        use_beam_search: bool = False,
        beam_width: int = 5
    ):
        """
        Args:
            checkpoint_path: Path to the model checkpoint
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
            use_beam_search: Use beam search instead of greedy decoding
            beam_width: Beam width for beam search
        """
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        if use_beam_search:
            logger.info(f"Beam search enabled with beam_width={beam_width}")

        # Load model
        self.model = None
        self.beam_search = None
        self.scalers = None
        self._load_model()

    def _load_model(self):
        """Load the model from a checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Create model
        self.model = AttentionVRP(
            node_features=12,  # Updated: 5 base + 7 order features
            edge_features=3,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.25,
            max_route_length=50
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Initialize beam search if enabled
        if self.use_beam_search:
            self.beam_search = BeamSearchVRP(self.model, beam_width=self.beam_width)

        logger.info(f"Model loaded successfully (val_acc={checkpoint['metrics']['accuracy']:.4f})")

    def predict_single(
        self,
        input_scenario: Dict,
        return_confidence: bool = False
    ) -> Dict:
        """
        Predict a route for a single scenario.

        Args:
            input_scenario: Input scenario (nodes, edges)
            return_confidence: Whether to return confidence scores

        Returns:
            Dict with the predicted route and metadata
        """
        start_time = time.time()

        try:
            # Create dataset (fit scalers if first time)
            if self.scalers is None:
                dataset = VRPNodeDataset([(input_scenario, {})], fit_scalers=True)
                self.scalers = (dataset.node_scaler, dataset.edge_scaler)
            else:
                dataset = VRPNodeDataset([(input_scenario, {})], fit_scalers=False)
                dataset.node_scaler = self.scalers[0]
                dataset.edge_scaler = self.scalers[1]

            graph_data = dataset[0].to(self.device)
            batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)

            # Inference
            with torch.no_grad():
                if self.use_beam_search:
                    # Use beam search
                    predicted_nodes = self.beam_search.search(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.edge_attr,
                        batch
                    )
                    logits = None  # Beam search doesn't return logits
                else:
                    # Use greedy decoding
                    logits, predictions, _ = self.model(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.edge_attr,
                        batch,
                        target_nodes=None,
                        teacher_forcing_ratio=0.0
                    )
                    predicted_nodes = predictions[0].cpu().tolist()

            inference_time = time.time() - start_time

            result = {
                'predicted_route': predicted_nodes,
                'num_nodes': len(predicted_nodes),
                'inference_time_sec': inference_time
            }

            if return_confidence:
                if self.use_beam_search:
                    # Beam search doesn't provide confidence scores
                    result['confidence_scores'] = None
                    result['avg_confidence'] = None
                    result['note'] = 'Confidence scores not available with beam search'
                else:
                    # Compute confidence scores from logits
                    probs = torch.softmax(logits[0], dim=-1)
                    max_probs = probs.max(dim=-1)[0].cpu().tolist()
                    result['confidence_scores'] = max_probs
                    result['avg_confidence'] = sum(max_probs) / len(max_probs)

            return result

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def predict_batch(
        self,
        input_scenarios: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Run inference over multiple scenarios.

        Args:
            input_scenarios: List of input scenarios
            show_progress: Whether to log progress

        Returns:
            List of results, one per scenario
        """
        results = []
        total = len(input_scenarios)

        logger.info(f"Processing {total} scenarios...")

        for i, scenario in enumerate(input_scenarios):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total} scenarios")

            result = self.predict_single(scenario, return_confidence=False)
            result['scenario_index'] = i
            results.append(result)

        logger.info(f"Batch processing complete: {total} scenarios")
        return results

    def predict_from_file(
        self,
        scenario_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Run inference for a scenario loaded from a file.

        Args:
            scenario_path: Path to the scenario JSON file
            output_path: Optional path to save the result

        Returns:
            Prediction result
        """
        import json

        # Load scenario
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenario = json.load(f)

        # Predict
        result = self.predict_single(scenario, return_confidence=True)
        result['input_file'] = scenario_path

        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {output_path}")

        return result
