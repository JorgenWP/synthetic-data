"""
Base classes for graph generation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseGraphGenerator(nn.Module, ABC):
    """
    Abstract base class for graph generation models.
    
    All graph generation models should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        num_samples: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate synthetic graph samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use for generation
            
        Returns:
            Generated graph tensor
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the loss for training.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Loss tensor
        """
        pass
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, filepath)
    
    def load_checkpoint(self, filepath: str, device: Optional[torch.device] = None) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            device: Device to load the model to
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
