import os
import sys
import pickle as cp
import networkx as nx
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm

# Import existing utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from bigg.data_process.data_util import get_graph_data
from bigg.common.configs import cmd_args


cmd_opt = argparse.ArgumentParser(description='Preprocess transaction data for BiGG')
cmd_opt.add_argument('-input_path', default='../../data/Transactions/raw/transactions_data.csv', 
                     help='Path to input file (.csv or .gpickle)')
cmd_opt.add_argument('-limit_size', type=int, default=None, 
                     help='Maximum number of nodes to keep')
cmd_opt.add_argument('-sampling_method', type=str, default='forest_fire', choices=['bfs', 'forest_fire'],
                     help='Method to downsample graph: "bfs" (dense core) or "forest_fire" (preserves sparsity)')

local_args, _ = cmd_opt.parse_known_args()

def bfs_sampling(G, target_size):
    """
    Snowball Sampling (BFS).
    Result: A very connected, dense subgraph. Good for connectivity, bad for sparsity.
    """
    print(f"Sampling (BFS) to {target_size} nodes...")
    start_node = max(dict(G.degree()).items(), key=lambda x: x[1])[0]
    
    sampled_nodes = set([start_node])
    queue = [start_node]
    
    while len(sampled_nodes) < target_size and queue:
        current = queue.pop(0)
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors) # Shuffle to avoid order bias
        
        for n in neighbors:
            if n not in sampled_nodes:
                sampled_nodes.add(n)
                queue.append(n)
                if len(sampled_nodes) >= target_size:
                    break
                    
    return G.subgraph(list(sampled_nodes)).copy()

def forest_fire_sampling(G, target_size, p=0.7):
    """
    Forest Fire Sampling.
    Result: A sparse, connected subgraph that preserves power-law properties.
    p: Forward burning probability (0.7 is a standard default for social/fin networks).
    """
    print(f"Sampling (Forest Fire) to {target_size} nodes with p={p}...")
    
    nodes = list(G.nodes())
    sampled_nodes = set()
    
    # Helper to pick a random seed not yet sampled
    def get_random_seed():
        candidates = [n for n in nodes if n not in sampled_nodes]
        return random.choice(candidates) if candidates else None

    # Start the fire
    while len(sampled_nodes) < target_size:
        seed = get_random_seed()
        if seed is None: break # No more nodes
        
        queue = [seed]
        sampled_nodes.add(seed)
        
        while queue and len(sampled_nodes) < target_size:
            current = queue.pop(0)
            
            # Get neighbors not yet visited in this burn
            # (Forest Fire normally is directed, but works on undirected as 'neighbors')
            neighbors = [n for n in G.neighbors(current) if n not in sampled_nodes]
            
            # "Geometric" selection: Keep burning with probability p
            # Effectively: sample X neighbors where X ~ Geometric(p)
            # Simplified: For each neighbor, burn with prob p/(1-p) or just p depending on formulation.
            # Here we use the Leskovec formulation: generate x ~ Geom(p), select x neighbors.
            
            if not neighbors:
                continue

            # Calculate how many neighbors to burn
            # Mean = p / (1-p). If p=0.7, mean is 2.33. 
            try:
                # Numpy geometric is number of trials to get success, so we adjust
                n_to_burn = np.random.geometric(1.0 - p) - 1
            except:
                n_to_burn = 1
            
            if n_to_burn > 0:
                random.shuffle(neighbors)
                targets = neighbors[:n_to_burn]
                
                for t in targets:
                    if t not in sampled_nodes:
                        sampled_nodes.add(t)
                        queue.append(t)
                        if len(sampled_nodes) >= target_size:
                            break
                            
    # Extract subgraph and remove isolates (Forest Fire naturally creates components)
    subgraph = G.subgraph(list(sampled_nodes)).copy()
    
    # Optional: Take largest connected component if BiGG strictly requires connectivity
    # (BiGG can handle components, but learning is more stable on the largest one)
    if nx.number_connected_components(subgraph) > 1:
        print("  - Note: Sampling created multiple components. Keeping largest component to ensure stability.")
        largest_cc = max(nx.connected_components(subgraph), key=len)
        subgraph = subgraph.subgraph(largest_cc).copy()

    return subgraph

def load_gpickle_structure(pickle_path, limit_size=None, method='forest_fire'):
    print(f"Loading graph from {pickle_path}...")
    try:
        with open(pickle_path, 'rb') as f:
            G_raw = cp.load(f)
    except:
        G_raw = nx.read_gpickle(pickle_path)

    # Structure only (Undirected)
    G_struct = nx.Graph()
    G_struct.add_nodes_from(G_raw.nodes())
    G_struct.add_edges_from(G_raw.edges())
    G_struct.remove_edges_from(nx.selfloop_edges(G_struct))

    if limit_size and G_struct.number_of_nodes() > limit_size:
        if method == 'bfs':
            G_struct = bfs_sampling(G_struct, limit_size)
        else:
            G_struct = forest_fire_sampling(G_struct, limit_size)
    
    # Relabel to 0...N-1
    G_struct = nx.convert_node_labels_to_integers(G_struct, ordering='sorted')
    return G_struct


if __name__ == '__main__':
    cmd_args.__dict__.update(local_args.__dict__)
    
    if not os.path.exists(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
    
    # Load the graph data and apply sampling if needed
    G = load_gpickle_structure(cmd_args.input_path, 
                                   limit_size=cmd_args.limit_size,
                                   method=cmd_args.sampling_method)
    
    
    print(f"Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"Ordering graph using {cmd_args.node_order}...")
    processed_graphs = get_graph_data(G, node_order=cmd_args.node_order)
    
    splits = {'train': processed_graphs, 'val': processed_graphs, 'test': processed_graphs}
    
    for phase, graphs in splits.items():
        save_path = os.path.join(cmd_args.save_dir, f'{phase}-graphs.pkl')
        with open(save_path, 'wb') as f:
            for g in graphs:
                cp.dump(g, f, cp.HIGHEST_PROTOCOL)
            
    print(f"Preprocessing complete!")