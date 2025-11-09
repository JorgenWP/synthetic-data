"""
Visualization utilities for graphs and training metrics.
"""

import os
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_graph(
    G: nx.Graph,
    save_path: Optional[str] = None,
    title: str = "Graph Visualization",
    figsize: tuple = (12, 8),
    node_size: int = 300,
    node_color: str = 'lightblue',
    with_labels: bool = True
) -> None:
    """
    Visualize a NetworkX graph.
    
    Args:
        G: NetworkX graph to visualize
        save_path: Optional path to save the figure
        title: Title for the plot
        figsize: Figure size
        node_size: Size of nodes
        node_color: Color of nodes
        with_labels: Whether to show node labels
    """
    plt.figure(figsize=figsize)
    
    # Choose layout based on graph size
    if G.number_of_nodes() < 100:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    if with_labels and G.number_of_nodes() < 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_degree_distribution(
    graphs: List[nx.Graph],
    save_path: Optional[str] = None,
    title: str = "Degree Distribution"
) -> None:
    """
    Plot degree distribution of graphs.
    
    Args:
        graphs: List of NetworkX graphs
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    all_degrees = []
    for g in graphs:
        all_degrees.extend([d for n, d in g.degree()])
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_degrees, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def compare_graph_statistics(
    real_stats: Dict[str, float],
    generated_stats: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Compare statistics between real and generated graphs.
    
    Args:
        real_stats: Statistics from real graphs
        generated_stats: Statistics from generated graphs
        save_path: Optional path to save the figure
    """
    metrics = list(real_stats.keys())
    real_values = [real_stats[m] for m in metrics]
    gen_values = [generated_stats[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, real_values, width, label='Real', alpha=0.8)
    plt.bar(x + width/2, gen_values, width, label='Generated', alpha=0.8)
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Graph Statistics Comparison', fontsize=14)
    plt.xticks(x, metrics, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
