"""
Tests for graph preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from src.data import GraphPreprocessor


class TestGraphPreprocessor:
    """Test cases for GraphPreprocessor class."""
    
    def test_initialization(self):
        """Test GraphPreprocessor initialization."""
        preprocessor = GraphPreprocessor()
        assert preprocessor.config == {}
        assert preprocessor.node_mapping == {}
        assert preprocessor.edge_types == []
    
    def test_build_graph(self):
        """Test graph building from transactions."""
        # Create sample transaction data
        df = pd.DataFrame({
            'source': ['A', 'B', 'C', 'A'],
            'target': ['B', 'C', 'D', 'C'],
            'amount': [100, 200, 150, 300]
        })
        
        preprocessor = GraphPreprocessor()
        G = preprocessor.build_graph_from_transactions(df, 'source', 'target', ['amount'])
        
        # Check graph properties
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4
        assert 'amount' in G['A']['B']
    
    def test_extract_adjacency_matrix(self):
        """Test adjacency matrix extraction."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        preprocessor = GraphPreprocessor()
        adj_matrix = preprocessor.extract_adjacency_matrix(G)
        
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix[0, 1] == 1
        assert adj_matrix[1, 2] == 1
        assert adj_matrix[2, 0] == 1
    
    def test_compute_graph_statistics(self):
        """Test graph statistics computation."""
        G = nx.complete_graph(5)
        
        preprocessor = GraphPreprocessor()
        stats = preprocessor.compute_graph_statistics(G)
        
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert stats['num_nodes'] == 5
        assert stats['num_edges'] == 10
        assert stats['density'] == 1.0
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.array([[1, 2], [3, 4], [5, 6]])
        
        preprocessor = GraphPreprocessor()
        
        # Test standard normalization
        normalized = preprocessor.normalize_features(features, method='standard')
        assert normalized.shape == features.shape
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6
        
        # Test minmax normalization
        normalized = preprocessor.normalize_features(features, method='minmax')
        assert normalized.shape == features.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1
