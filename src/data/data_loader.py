"""
Data loader for VRP scenarios with input/output JSON files.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VRPDataLoader:
    """Loads and pairs input/output JSON files for VRP scenarios."""

    def __init__(self, dataset_root: str):
        """
        Initialize the data loader.

        Args:
            dataset_root: Root directory containing input and output folders
        """
        self.dataset_root = Path(dataset_root)
        self.input_folders = []
        self.output_folders = []
        self._discover_folders()

    def _discover_folders(self):
        """Discover all input and output folders in the dataset."""
        for item in self.dataset_root.iterdir():
            if item.is_dir():
                if 'input' in item.name.lower():
                    self.input_folders.append(item)
                # Output folders can be named "1-5000", "5001-10000" etc (without "output" prefix)
                # or "output 1-5000" etc (with "output" prefix)
                elif 'output' in item.name.lower() or item.name.replace('-', '').isdigit():
                    # Check if folder contains RESULT_*.json files (output format)
                    json_files = list(item.glob('*.json')) + list(item.glob('*/*.json'))
                    if any('RESULT_' in f.name for f in json_files):
                        self.output_folders.append(item)

        self.input_folders.sort()
        self.output_folders.sort()
        logger.info(f"Found {len(self.input_folders)} input folders and {len(self.output_folders)} output folders")

    def _find_nested_json_files(self, folder: Path) -> List[Path]:
        """Find all JSON files in folder and subfolders."""
        json_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(Path(root) / file)
        return sorted(json_files)

    def load_scenario_pairs(self, max_samples: Optional[int] = None, extract_route_only: bool = True) -> List[Tuple[Dict, Dict]]:
        """
        Load all scenario pairs (input, output).

        Args:
            max_samples: Maximum number of samples to load (None = all)
            extract_route_only: If True, extract only route indices (V1 format).
                               If False, keep full output scenario (V2 format).

        Returns:
            List of tuples (input_data, output_data)
        """
        pairs = []

        # Build output file index with folder name to avoid collisions
        logger.info("Building output file index...")
        output_index = {}
        for output_folder in self.output_folders:
            output_files = self._find_nested_json_files(output_folder)
            folder_name = output_folder.name  # e.g., "output 1-5000"
            for output_file in output_files:
                # Extract scenario_id from RESULT_scenario_XXXX.json
                scenario_id = output_file.stem.replace('RESULT_', '')
                # Use (folder_name, scenario_id) as key to avoid collisions
                key = (folder_name, scenario_id)
                output_index[key] = output_file

        logger.info(f"Found {len(output_index)} output files")

        # Collect all input files with their folder names
        logger.info("Collecting input files...")
        all_input_files = []
        for input_folder in self.input_folders:
            input_files = self._find_nested_json_files(input_folder)
            folder_name = input_folder.name  # e.g., "input 1-5000"
            for input_file in input_files:
                all_input_files.append((folder_name, input_file))

        logger.info(f"Found {len(all_input_files)} input files")

        # Limit samples if requested
        if max_samples:
            all_input_files = all_input_files[:max_samples]
            logger.info(f"Limited to {max_samples} samples")

        # Load pairs with progress bar
        logger.info("Loading scenario pairs...")
        for folder_name, input_file in tqdm(all_input_files, desc="Loading data"):
            scenario_id = input_file.stem

            # Try multiple output folder name patterns
            # Pattern 1: "input 1-5000" -> "output 1-5000"
            output_folder_name1 = folder_name.replace('input', 'output')
            # Pattern 2: "input 1-5000" -> "1-5000" (remove "input " prefix)
            output_folder_name2 = folder_name.replace('input ', '')

            # Try both patterns
            key1 = (output_folder_name1, scenario_id)
            key2 = (output_folder_name2, scenario_id)

            output_file = output_index.get(key1) or output_index.get(key2)

            if output_file:
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                    with open(output_file, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)

                    # Extract route from nested structure (V1) or keep full output (V2)
                    if extract_route_only:
                        output_data = self._extract_route(output_data, input_data)

                    pairs.append((input_data, output_data))
                except Exception as e:
                    logger.warning(f"Failed to load {input_file.name}: {e}")

        logger.info(f"Loaded {len(pairs)} scenario pairs")
        return pairs

    def _extract_route(self, output_data: dict, input_data: dict) -> dict:
        """Extract route from nested structure and convert edge IDs to indices."""
        # Build edge ID to index mapping
        edge_id_to_idx = {}
        for idx, edge in enumerate(input_data.get('edges', [])):
            edge_id = edge.get('edge_id', f'edge_{idx}')
            edge_id_to_idx[edge_id] = idx

        # Extract route from vehicleTypeToOptimalRouteList
        route_indices = []
        vehicle_routes = output_data.get('vehicleTypeToOptimalRouteList', {})

        for vehicle_type, routes in vehicle_routes.items():
            if routes and len(routes) > 0:
                route_edge_ids = routes[0].get('route', [])
                for edge_id in route_edge_ids:
                    if edge_id in edge_id_to_idx:
                        route_indices.append(edge_id_to_idx[edge_id])

        return {'route': route_indices}


    def _find_matching_output(self, scenario_id: str) -> Optional[Path]:
        """Find the output file matching the given scenario_id."""
        for output_folder in self.output_folders:
            output_files = self._find_nested_json_files(output_folder)
            for output_file in output_files:
                if scenario_id in output_file.stem:
                    return output_file
        return None
