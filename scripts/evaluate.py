#!/usr/bin/env python3
"""
Evaluation script for trained graph generation models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default_config.yaml
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models import GraphVAE
from src.training import Evaluator


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate graph generation model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to generate for evaluation'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(
        config['hardware']['device']
        if torch.cuda.is_available() and config['hardware']['device'] == 'cuda'
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model_config = config['model']
    if model_config['type'] == 'GraphVAE':
        model = GraphVAE(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model.load_checkpoint(args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        generated_samples = model.generate(args.num_samples, device)
    
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # TODO: Convert generated samples to graph structures and evaluate
    print("\n" + "="*80)
    print("NOTE: Evaluation metrics need to be implemented based on your specific task.")
    print("Please modify this script to:")
    print("1. Convert generated samples to graph structures")
    print("2. Load real graphs for comparison")
    print("3. Compute evaluation metrics")
    print("="*80 + "\n")
    
    # Example structure for when evaluation is ready:
    # evaluator = Evaluator()
    # real_graphs = load_real_graphs(config)
    # generated_graphs = convert_to_graphs(generated_samples)
    # metrics = evaluator.compare_distributions(real_graphs, generated_graphs)
    # print("Evaluation Results:")
    # for metric, value in metrics.items():
    #     print(f"  {metric}: {value:.4f}")
    
    print("Evaluation script template ready. Implement graph conversion and metrics.")


if __name__ == "__main__":
    main()
