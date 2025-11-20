import os
import sys
import pickle as cp
import networkx as nx
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# Import existing utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from bigg.data_process.data_util import get_graph_data
from bigg.common.configs import cmd_args  # Import the shared config parser

# Define local arguments specific to this script
cmd_opt = argparse.ArgumentParser(description='Preprocess transaction data for BiGG')
cmd_opt.add_argument('-csv_path', default='../../data/Transactions/raw/transactions_data.csv', help='Path to input CSV')
cmd_opt.add_argument('-nrows', default=None, type=int, help='Limit number of rows (e.g., 10000 for testing)')

# Parse local arguments
local_args, _ = cmd_opt.parse_known_args()

def load_transaction_graph_structure(csv_path, limit=None):
    """
    Reads the CSV and constructs a NetworkX graph (Structure Only).
    """
    print(f"Reading {'all' if limit is None else str(limit)} rows from CSV...")
    df = pd.read_csv(csv_path, nrows=limit)
    
    # Map IDs to unique Integers
    # Prefix IDs to ensure Client 123 is distinct from Merchant 123
    df['src_node'] = 'c_' + df['client_id'].astype(str)
    df['dst_node'] = 'm_' + df['merchant_id'].astype(str)
    
    unique_nodes = pd.concat([df['src_node'], df['dst_node']]).unique()
    node_map = {node_id: i for i, node_id in enumerate(unique_nodes)}
    
    df['u'] = df['src_node'].map(node_map)
    df['v'] = df['dst_node'].map(node_map)
    
    # Build Graph
    G = nx.Graph()
    G.add_nodes_from(node_map.values())
    
    # Add edges (using list of tuples for speed)
    edges = list(zip(df['u'], df['v']))
    G.add_edges_from(edges)
                   
    return G

if __name__ == '__main__':
    # Merge local arguments into the global cmd_args (overwriting defaults if provided)
    cmd_args.__dict__.update(local_args.__dict__)
    
    # Configuration from arguments
    # cmd_args.save_dir, cmd_args.node_order come from bigg.common.configs
    # cmd_args.csv_path, cmd_args.nrows come from local parser above
    
    if not os.path.exists(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
        print(f"Created directory: {cmd_args.save_dir}")
        
    # 1. Load Graph
    print(f"Loading CSV from {cmd_args.csv_path}...")
    G = load_transaction_graph_structure(cmd_args.csv_path, limit=cmd_args.nrows)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. Process and Order
    print(f"Ordering graph using {cmd_args.node_order}...")
    processed_graphs = get_graph_data(G, node_order=cmd_args.node_order)
    
    print(f"Generated {len(processed_graphs)} component(s).")

    # 3. Assign to Splits (Single Train Set Strategy)
    # As discussed, we replicate the graph across train/val/test to satisfy pipeline requirements.
    print("Assigning entire dataset to Train, Val, and Test splits...")
    
    splits = {
        'train': processed_graphs,
        'val': processed_graphs,
        'test': processed_graphs
    }
    
    # 4. Save to Disk
    print(f"Saving files to {cmd_args.save_dir}...")
    for phase, graphs in splits.items():
        save_path = os.path.join(cmd_args.save_dir, f'{phase}-graphs.pkl')
        print(f"  - Saving {len(graphs)} graph(s) to {os.path.basename(save_path)}")
        with open(save_path, 'wb') as f:
            for g in graphs:
                cp.dump(g, f, cp.HIGHEST_PROTOCOL)
            
    print(f"Preprocessing complete! Config: Order={cmd_args.node_order}, Rows={cmd_args.nrows}")