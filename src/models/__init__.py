"""
Model architectures for graph generation.
"""

from .base import BaseGraphGenerator
from .vae import GraphVAE

__all__ = ["BaseGraphGenerator", "GraphVAE"]
