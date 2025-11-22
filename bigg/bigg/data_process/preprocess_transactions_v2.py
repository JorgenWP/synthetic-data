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

# Define Arguments
cmd_opt = argparse.ArgumentParser(description='Preprocess transaction data for BiGG')
cmd_opt.add_argument('-input_path', default='../../data/Transactions/raw/transactions_data.csv', 
                     help='Path to input file (.csv or .gpickle)')
cmd_opt.add_argument('-save_dir', default='../../data/Transactions/processed_multigraph', 
                     help='Directory to save output')
cmd_opt.add_argument('-node_order', default='DFS', choices=['DFS', 'BFS'], help='Traversal ordering for BiGG')

# Sampling Arguments
cmd_opt.add_argument('-num_graphs', type=int, default=50, 
                     help='Number of subgraphs to extract')
cmd_opt.add_argument('-min_nodes', type=int, default=500, 
                     help='Minimum nodes per subgraph')
cmd_opt.add_argument('-max_nodes', type=int, default=1000, 
                     help='Maximum nodes per subgraph')
cmd_opt.add_argument('-sampling_method', type=str, default='forest_fire', choices=['bfs', 'forest_fire'],
                     help='Method to downsample graph')

local_args, _ = cmd_opt.parse_known_args()

def load_graph_structure(file_path):
    """
    Loads the full raw graph into memory with ORIGINAL labels.
    Does NOT convert to integers yet to save RAM.
    """
    print(f"Loading full graph from {file_path}...")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Ensure unique IDs for mapping
        df['src'] = 'c_' + df['client_id'].astype(str)
        df['dst'] = 'm_' + df['merchant_id'].astype(str)
        
        G = nx.Graph()
        # NetworkX handles string labels perfectly fine
        G.add_edges_from(zip(df['src'], df['dst']))
        
    else: # Pickle / Gpickle
        try:
            with open(file_path, 'rb') as f:
                G_raw = cp.load(f)
        except:
            G_raw = nx.read_gpickle(file_path)
        
        G = nx.Graph()
        G.add_edges_from(G_raw.edges())
        G.remove_edges_from(nx.selfloop_edges(G))

    print(f"  - Full Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def bfs_sampling(G, target_size):
    """
    Randomized BFS Sampling on the original graph.
    """
    # Pick random node. Since labels are strings, we convert keys to list.
    nodes = list(G.nodes())
    
    # Optimization: Calculating degrees for 1.6M nodes can be slow.
    # Uniform random choice is much faster and sufficient for random patches.
    # If you really need degree-weighted:
    degrees = np.array([d for n, d in G.degree()])
    probs = degrees / degrees.sum()
    start_node = np.random.choice(nodes, p=probs)
    
    # start_node = random.choice(nodes)
    
    sampled_nodes = set([start_node])
    queue = [start_node]
    
    while len(sampled_nodes) < target_size and queue:
        current = queue.pop(0)
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)
        
        for n in neighbors:
            if n not in sampled_nodes:
                sampled_nodes.add(n)
                queue.append(n)
                if len(sampled_nodes) >= target_size:
                    break
                    
    return G.subgraph(list(sampled_nodes)).copy()

def forest_fire_sampling(G, target_size, p=0.7):
    """
    Forest Fire Sampling on the original graph.
    """
    nodes = list(G.nodes())
    sampled_nodes = set()
    
    def get_random_seed():
        candidates = [n for n in nodes if n not in sampled_nodes]
        if not candidates: candidates = nodes
        return random.choice(candidates)

    # Safety counter to prevent infinite loops on disconnected graphs
    attempts = 0
    max_attempts = target_size * 2

    while len(sampled_nodes) < target_size and attempts < max_attempts:
        seed = get_random_seed()
        queue = [seed]
        sampled_nodes.add(seed)
        attempts += 1
        
        while queue and len(sampled_nodes) < target_size:
            current = queue.pop(0)
            neighbors = [n for n in G.neighbors(current) if n not in sampled_nodes]
            
            try:
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
                            
    subgraph = G.subgraph(list(sampled_nodes)).copy()
    
    if not nx.is_connected(subgraph):
        largest_cc = max(nx.connected_components(subgraph), key=len)
        subgraph = subgraph.subgraph(largest_cc).copy()

    return subgraph

if __name__ == '__main__':
    cmd_args.__dict__.update(local_args.__dict__)

    if not os.path.exists(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

    # Load the Graph 
    full_graph = load_graph_structure(cmd_args.input_path)
    
    # Extract Subgraphs
    all_subgraphs = []
    print(f"\nGenerating {cmd_args.num_graphs} subgraphs ({cmd_args.min_nodes}-{cmd_args.max_nodes} nodes)...")
    
    for i in tqdm(range(cmd_args.num_graphs)):
        target_size = random.randint(cmd_args.min_nodes, cmd_args.max_nodes)
        
        if cmd_args.sampling_method == 'bfs':
            subG = bfs_sampling(full_graph, target_size)
        else:
            subG = forest_fire_sampling(full_graph, target_size)
        
        # Convert node labels to integers
        subG = nx.convert_node_labels_to_integers(subG)
        
        # Process for BiGG (Canonical Ordering)
        processed_parts = get_graph_data(subG, node_order=cmd_args.node_order)
        if processed_parts:
            all_subgraphs.append(processed_parts[0])

    print(f"Successfully created {len(all_subgraphs)} subgraphs.")

    # Split Data
    random.shuffle(all_subgraphs)
    n_train = int(len(all_subgraphs) * 0.8)
    n_val = int(len(all_subgraphs) * 0.1)
    
    splits = {
        'train': all_subgraphs[:n_train],
        'val': all_subgraphs[n_train:n_train+n_val],
        'test': all_subgraphs[n_train+n_val:]
    }
    
    if not splits['val'] and splits['train']: splits['val'] = [splits['train'][0]]
    if not splits['test'] and splits['train']: splits['test'] = [splits['train'][0]]

    # Save
    for phase, graphs in splits.items():
        save_path = os.path.join(cmd_args.save_dir, f'{phase}-graphs.pkl')
        print(f"Saving {len(graphs)} graphs to {save_path}...")
        with open(save_path, 'wb') as f:
            for g in graphs:
                cp.dump(g, f, cp.HIGHEST_PROTOCOL)
            
    print(f"\nPreprocessing complete! Data saved to {cmd_args.save_dir}")