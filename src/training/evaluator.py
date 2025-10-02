"""
Evaluation metrics for generated graphs.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import networkx as nx
from scipy.stats import ks_2samp


class Evaluator:
    """
    Evaluate quality of generated graphs.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def compute_graph_statistics(self, graphs: List[nx.Graph]) -> Dict[str, Any]:
        """
        Compute statistics for a list of graphs.
        
        Args:
            graphs: List of NetworkX graphs
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_graphs': len(graphs),
            'avg_nodes': np.mean([g.number_of_nodes() for g in graphs]),
            'avg_edges': np.mean([g.number_of_edges() for g in graphs]),
            'avg_density': np.mean([nx.density(g) for g in graphs if g.number_of_nodes() > 0]),
        }
        
        # Degree statistics
        all_degrees = []
        for g in graphs:
            all_degrees.extend([d for n, d in g.degree()])
        
        if all_degrees:
            stats['avg_degree'] = np.mean(all_degrees)
            stats['std_degree'] = np.std(all_degrees)
        
        return stats
    
    def compare_distributions(
        self,
        real_graphs: List[nx.Graph],
        generated_graphs: List[nx.Graph],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compare distributions of real and generated graphs.
        
        Args:
            real_graphs: List of real graphs
            generated_graphs: List of generated graphs
            metrics: List of metrics to compare (default: ['degree', 'clustering'])
            
        Returns:
            Dictionary of comparison scores (lower is better for KS statistic)
        """
        if metrics is None:
            metrics = ['degree', 'clustering']
        
        results = {}
        
        for metric in metrics:
            if metric == 'degree':
                real_vals = self._extract_degree_sequence(real_graphs)
                gen_vals = self._extract_degree_sequence(generated_graphs)
            elif metric == 'clustering':
                real_vals = self._extract_clustering_coefficients(real_graphs)
                gen_vals = self._extract_clustering_coefficients(generated_graphs)
            else:
                continue
            
            # Kolmogorov-Smirnov test
            if len(real_vals) > 0 and len(gen_vals) > 0:
                ks_stat, p_value = ks_2samp(real_vals, gen_vals)
                results[f'{metric}_ks_stat'] = ks_stat
                results[f'{metric}_p_value'] = p_value
        
        return results
    
    def _extract_degree_sequence(self, graphs: List[nx.Graph]) -> np.ndarray:
        """Extract degree sequence from graphs."""
        degrees = []
        for g in graphs:
            degrees.extend([d for n, d in g.degree()])
        return np.array(degrees)
    
    def _extract_clustering_coefficients(self, graphs: List[nx.Graph]) -> np.ndarray:
        """Extract clustering coefficients from graphs."""
        coeffs = []
        for g in graphs:
            clustering = nx.clustering(g)
            coeffs.extend(clustering.values())
        return np.array(coeffs)
    
    def evaluate_diversity(self, graphs: List[nx.Graph]) -> float:
        """
        Measure diversity of generated graphs.
        
        Args:
            graphs: List of generated graphs
            
        Returns:
            Diversity score (higher is better)
        """
        # Simple diversity measure based on graph edit distance or structural differences
        # This is a placeholder - implement based on your specific needs
        if len(graphs) < 2:
            return 0.0
        
        # Example: measure variance in basic statistics
        num_nodes = [g.number_of_nodes() for g in graphs]
        num_edges = [g.number_of_edges() for g in graphs]
        
        diversity = np.std(num_nodes) + np.std(num_edges)
        return float(diversity)
    
    def compute_mmd(
        self,
        real_features: np.ndarray,
        generated_features: np.ndarray,
        kernel: str = 'rbf'
    ) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between real and generated features.
        
        Args:
            real_features: Features from real graphs
            generated_features: Features from generated graphs
            kernel: Kernel type ('rbf', 'linear')
            
        Returns:
            MMD score (lower is better)
        """
        # Simple MMD implementation
        # For production, consider using more sophisticated implementations
        
        def kernel_matrix(x, y, kernel_type='rbf'):
            if kernel_type == 'rbf':
                gamma = 1.0 / x.shape[1]
                xx = np.sum(x ** 2, axis=1)[:, np.newaxis]
                yy = np.sum(y ** 2, axis=1)[np.newaxis, :]
                xy = np.dot(x, y.T)
                return np.exp(-gamma * (xx + yy - 2 * xy))
            elif kernel_type == 'linear':
                return np.dot(x, y.T)
            else:
                raise ValueError(f"Unknown kernel: {kernel_type}")
        
        k_xx = kernel_matrix(real_features, real_features, kernel)
        k_yy = kernel_matrix(generated_features, generated_features, kernel)
        k_xy = kernel_matrix(real_features, generated_features, kernel)
        
        m = real_features.shape[0]
        n = generated_features.shape[0]
        
        mmd = (k_xx.sum() / (m * m) + 
               k_yy.sum() / (n * n) - 
               2 * k_xy.sum() / (m * n))
        
        return float(mmd)
