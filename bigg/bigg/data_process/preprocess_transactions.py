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

DATASET_START_DATE = '2010-01-01'
DATASET_END_DATE = '2019-11-01'

# Define local arguments specific to this script
cmd_opt = argparse.ArgumentParser(description='Preprocess transaction data for BiGG')
cmd_opt.add_argument('-csv_path', default='../../data/Transactions/raw/transactions_data.csv', help='Path to input CSV')
cmd_opt.add_argument('-start_date', default=DATASET_START_DATE, help='Start date for transactions (YYYY-MM-DD)')
cmd_opt.add_argument('-cutoff_date', default=DATASET_END_DATE, help='Cutoff date for transactions (YYYY-MM-DD)')


# Parse local arguments
local_args, _ = cmd_opt.parse_known_args()

def load_transaction_graph_structure(csv_path, start_date, cutoff_date):
    """
    Reads the CSV and constructs a NetworkX graph (Structure Only).
    """
    print(f"Reading data from CSV...")
    df = pd.read_csv(csv_path)

    print(f"Filtering data from {start_date} to {cutoff_date}...")
    # Filter rows after a given date
    df['date'] = pd.to_datetime(df['date']) # Ensure date column is datetime
    df = df[(df['date'] >= start_date) & (df['date'] <= cutoff_date)] # Keep rows within the date range
    
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
    # cmd_args.csv_path, cmd_args.cutoff_date, cmd_args.start_date come from local parser above

    # Date validation
    if cmd_args.start_date > cmd_args.cutoff_date:
        raise ValueError("Start date must be earlier than or equal to cutoff date.")
    if cmd_args.start_date < DATASET_START_DATE:
        raise ValueError(f"Start date must be on or after {DATASET_START_DATE}.")
    if cmd_args.cutoff_date > DATASET_END_DATE:
        raise ValueError(f"Cutoff date must be on or before {DATASET_END_DATE}.")
    
    if not os.path.exists(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
        print(f"Created directory: {cmd_args.save_dir}")
        
    # 1. Load Graph
    print(f"Loading CSV from {cmd_args.csv_path}...")
    G = load_transaction_graph_structure(cmd_args.csv_path, start_date=cmd_args.start_date, cutoff_date=cmd_args.cutoff_date)
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
            
    print(f"Preprocessing complete! Config: Order={cmd_args.node_order}, Start Date={cmd_args.start_date}, Cutoff Date={cmd_args.cutoff_date}")