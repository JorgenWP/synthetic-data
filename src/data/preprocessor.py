"""
Graph preprocessing utilities for transactional data.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import networkx as nx


class GraphPreprocessor:
    """
    Convert transactional data into graph representations.
    
    This class provides methods to transform raw transactional data
    into graph structures suitable for training generative models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing
        """
        self.config = config or {}
        self.node_mapping = {}
        self.edge_types = []
        
    def build_graph_from_transactions(
        self,
        df: pd.DataFrame,
        source_col: str = "source",
        target_col: str = "target",
        edge_attrs: Optional[List[str]] = None
    ) -> nx.Graph:
        """
        Build a NetworkX graph from transactional data.
        
        Args:
            df: DataFrame containing transactional data
            source_col: Name of the source node column
            target_col: Name of the target node column
            edge_attrs: List of column names to use as edge attributes
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        for idx, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]
            
            # Add edge with optional attributes
            edge_data = {}
            if edge_attrs:
                for attr in edge_attrs:
                    if attr in row:
                        edge_data[attr] = row[attr]
            
            G.add_edge(source, target, **edge_data)
        
        return G
    
    def extract_node_features(
        self,
        G: nx.Graph,
        feature_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract node features from the graph.
        
        Args:
            G: NetworkX graph
            feature_list: List of node features to extract
            
        Returns:
            Array of node features
        """
        features = []
        
        for node in G.nodes():
            node_features = []
            
            # Basic structural features
            node_features.append(G.degree(node))
            
            # Add custom features if specified
            if feature_list:
                for feat in feature_list:
                    if feat in G.nodes[node]:
                        node_features.append(G.nodes[node][feat])
            
            features.append(node_features)
        
        return np.array(features)
    
    def extract_adjacency_matrix(self, G: nx.Graph) -> np.ndarray:
        """
        Extract adjacency matrix from graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Adjacency matrix as numpy array
        """
        return nx.to_numpy_array(G)
    
    def compute_graph_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute statistics about the graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        }
        
        # Add more statistics if the graph is connected and not too large
        if G.number_of_nodes() > 0 and G.number_of_nodes() < 1000:
            if nx.is_connected(G):
                stats["diameter"] = nx.diameter(G)
                stats["avg_path_length"] = nx.average_shortest_path_length(G)
        
        return stats
    
    def normalize_features(
        self,
        features: np.ndarray,
        method: str = "standard"
    ) -> np.ndarray:
        """
        Normalize node features.
        
        Args:
            features: Array of features to normalize
            method: Normalization method ('standard', 'minmax')
            
        Returns:
            Normalized features
        """
        if method == "standard":
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            return (features - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val = features.min(axis=0)
            max_val = features.max(axis=0)
            return (features - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
