#!/usr/bin/env python3
"""
Training script for graph generation models.

Usage:
    python scripts/train.py --config configs/default_config.yaml
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models import GraphVAE
from src.training import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train graph generation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Setup device
    device = torch.device(
        config['hardware']['device']
        if torch.cuda.is_available() and config['hardware']['device'] == 'cuda'
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model_config = config['model']
    if model_config['type'] == 'GraphVAE':
        model = GraphVAE(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create trainer
    training_config = {
        **config['training'],
        **config['experiment']
    }
    trainer = Trainer(model, optimizer, device, training_config)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # TODO: Create data loaders
    # For now, this is a placeholder - you need to implement your data loading
    print("\n" + "="*80)
    print("NOTE: Data loaders need to be implemented based on your specific data format.")
    print("Please modify this script to load your transactional data and create")
    print("appropriate PyTorch DataLoader objects for train_loader and val_loader.")
    print("="*80 + "\n")
    
    # Example structure for when data loaders are ready:
    # train_loader = create_dataloader(config, split='train')
    # val_loader = create_dataloader(config, split='val')
    # trainer.train(train_loader, val_loader)
    
    print("Training script template ready. Implement data loading to start training.")


if __name__ == "__main__":
    main()
