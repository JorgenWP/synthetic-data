"""
Variational Autoencoder for graph generation.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGraphGenerator


class GraphVAE(BaseGraphGenerator):
    """
    Variational Autoencoder for generating graph structures.
    
    This is a template implementation that can be customized
    for specific graph generation tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphVAE model.
        
        Args:
            config: Configuration dictionary with keys:
                - input_dim: Dimension of input features
                - hidden_dim: Dimension of hidden layers
                - latent_dim: Dimension of latent space
                - num_nodes: Expected number of nodes (for generation)
        """
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.latent_dim = config.get('latent_dim', 64)
        self.num_nodes = config.get('num_nodes', 100)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mu, logvar) for the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            predictions: Tuple of (reconstruction, mu, logvar)
            targets: Ground truth targets
            beta: Weight for KL divergence term
            
        Returns:
            Total loss
        """
        reconstruction, mu, logvar = predictions
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, targets, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss
    
    def generate(
        self,
        num_samples: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate synthetic graph samples from random latent vectors.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use for generation
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior (standard normal)
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode to generate samples
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples
