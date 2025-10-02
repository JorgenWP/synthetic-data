"""
Tests for model architectures.
"""

import pytest
import torch
from src.models import GraphVAE


class TestGraphVAE:
    """Test cases for GraphVAE model."""
    
    @pytest.fixture
    def model_config(self):
        """Create a model configuration for testing."""
        return {
            'input_dim': 64,
            'hidden_dim': 128,
            'latent_dim': 32,
            'num_nodes': 50
        }
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = GraphVAE(model_config)
        assert model.input_dim == 64
        assert model.hidden_dim == 128
        assert model.latent_dim == 32
    
    def test_forward_pass(self, model_config):
        """Test forward pass through the model."""
        model = GraphVAE(model_config)
        batch_size = 8
        x = torch.randn(batch_size, model_config['input_dim'])
        
        reconstruction, mu, logvar = model(x)
        
        assert reconstruction.shape == (batch_size, model_config['input_dim'])
        assert mu.shape == (batch_size, model_config['latent_dim'])
        assert logvar.shape == (batch_size, model_config['latent_dim'])
    
    def test_encode(self, model_config):
        """Test encoding functionality."""
        model = GraphVAE(model_config)
        x = torch.randn(4, model_config['input_dim'])
        
        mu, logvar = model.encode(x)
        
        assert mu.shape == (4, model_config['latent_dim'])
        assert logvar.shape == (4, model_config['latent_dim'])
    
    def test_decode(self, model_config):
        """Test decoding functionality."""
        model = GraphVAE(model_config)
        z = torch.randn(4, model_config['latent_dim'])
        
        output = model.decode(z)
        
        assert output.shape == (4, model_config['input_dim'])
    
    def test_generate(self, model_config):
        """Test generation functionality."""
        model = GraphVAE(model_config)
        num_samples = 10
        
        samples = model.generate(num_samples)
        
        assert samples.shape == (num_samples, model_config['input_dim'])
    
    def test_compute_loss(self, model_config):
        """Test loss computation."""
        model = GraphVAE(model_config)
        batch_size = 8
        x = torch.randn(batch_size, model_config['input_dim'])
        
        predictions = model(x)
        loss = model.compute_loss(predictions, x)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
    
    def test_count_parameters(self, model_config):
        """Test parameter counting."""
        model = GraphVAE(model_config)
        num_params = model.count_parameters()
        
        assert num_params > 0
        assert isinstance(num_params, int)
