import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from networkx.algorithms import community


DATA_BASE_DIR = "../../data/Transactions/transactions-BFS"


PATHS = {
    "Generated": "../../results/transactions/BFS-blksize--1-b-5/epoch-50.ckpt.graphs-0",
    "Training":  os.path.join(DATA_BASE_DIR, "train-graphs.pkl"),
    "Validation": os.path.join(DATA_BASE_DIR, "val-graphs.pkl"),
    "Test":       os.path.join(DATA_BASE_DIR, "test-graphs.pkl")
}

OUTPUT_DIR = "../../results/stats_output"

def load_graphs(path):
    print(f"Loading graphs from {path}...")
    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path}")
        return []
        
    with open(path, 'rb') as f:
        try:
            graphs = pickle.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to load pickle: {e}")
            return []
            
    if isinstance(graphs, list):
        return graphs
    return [graphs]

def get_graph_stats(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0: return None, []

    stats = {}
    stats['nodes'] = n
    stats['edges'] = m
    
    degrees = [d for _, d in G.degree()]
    stats['avg_degree'] = np.mean(degrees) if degrees else 0
    
    stats['density'] = nx.density(G)
    
    stats['components'] = nx.number_connected_components(G)
    
    try:
        communities = list(community.greedy_modularity_communities(G))
        stats['num_communities'] = len(communities)
        stats['modularity'] = community.modularity(G, communities)
    except Exception:
        stats['num_communities'] = 0
        stats['modularity'] = 0

    if n < 5000:
        try:
            if nx.is_connected(G):
                G_calc = G
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                G_calc = G.subgraph(largest_cc)
            
            stats['diameter'] = nx.diameter(G_calc)
            stats['avg_path_len'] = nx.average_shortest_path_length(G_calc)
        except Exception:
            stats['diameter'] = -1
            stats['avg_path_len'] = -1
    else:
        stats['diameter'] = -1
        stats['avg_path_len'] = -1
        
    return stats, degrees

def process_dataset(graphs, label):
    print(f"\n--- Processing {label} ({len(graphs)} graphs) ---")
    if not graphs: return None, [], []

    all_stats = []
    all_degrees = []
    all_clustering = []
    
    for i, G in enumerate(graphs):
        if i % 5 == 0: print(f"  Graph {i}/{len(graphs)}...")
        stats, degrees = get_graph_stats(G)
        if stats:
            all_stats.append(stats)
            all_degrees.extend(degrees)
            if G.number_of_nodes() < 5000:
                all_clustering.extend(list(nx.clustering(G).values()))
    
    avg_stats = {}
    if all_stats:
        for key in all_stats[0].keys():
            vals = [s[key] for s in all_stats if s[key] != -1]
            avg_stats[key] = np.mean(vals) if vals else 0
            
    return avg_stats, all_degrees, all_clustering

def plot_comparison(data_dict, title, xlabel, filename, log_scale=False):
    plt.figure(figsize=(12, 7))
    
    colors = {'Generated': 'orange', 'Training': 'blue', 'Validation': 'green', 'Test': 'red'}
    styles = {'Generated': 'bar', 'Training': 'bar', 'Validation': 'step', 'Test': 'step'}
    alphas = {'Generated': 0.5, 'Training': 0.4, 'Validation': 1.0, 'Test': 1.0}
    
    for name in ['Training', 'Generated']:
        if name in data_dict and data_dict[name]:
            plt.hist(data_dict[name], bins=50, density=True, alpha=alphas[name], 
                     label=name, color=colors[name], histtype='bar', edgecolor='black')

    for name in ['Validation', 'Test']:
        if name in data_dict and data_dict[name]:
            plt.hist(data_dict[name], bins=50, density=True, linewidth=2,
                     label=name, color=colors[name], histtype='step')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Probability Density")
    if log_scale: plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = {}
    for name, path in PATHS.items():
        graphs = load_graphs(path)
        avgs, degrees, clust = process_dataset(graphs, name)
        if avgs:
            results[name] = {
                'avgs': avgs,
                'degrees': degrees,
                'clustering': clust
            }

    print("\n" + "="*100)
    headers = ["METRIC"] + list(results.keys())
    header_str = f"{headers[0]:<20} | " + " | ".join([f"{h:<15}" for h in headers[1:]])
    print(header_str)
    print("-" * 100)
    
    metrics = [
        ('nodes', 'Avg Nodes'),
        ('edges', 'Avg Edges'),
        ('avg_degree', 'Avg Degree'),
        ('density', 'Density'),
        ('components', 'Components'),
        ('modularity', 'Modularity'),
        ('num_communities', 'Communities'),
        ('diameter', 'Diameter'),
        ('avg_path_len', 'Avg Path Len')
    ]
    
    for key, label in metrics:
        row_str = f"{label:<20} | "
        for name in headers[1:]:
            if name in results:
                val = results[name]['avgs'].get(key, 0)
                row_str += f"{val:<15.4f} | "
            else:
                row_str += f"{'N/A':<15} | "
        print(row_str)
    print("="*100)

    if 'Generated' in results:
        print("\nGenerating plots...")
        degree_data = {name: res['degrees'] for name, res in results.items()}
        clust_data = {name: res['clustering'] for name, res in results.items()}
        
        plot_comparison(degree_data, "Degree Distribution Comparison", "Degree", "comp_degree_full.png", log_scale=True)
        plot_comparison(clust_data, "Clustering Coeff Comparison", "Clustering Coeff", "comp_clustering_full.png")

if __name__ == "__main__":
    main()