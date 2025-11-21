import os
import sys
import argparse
import pickle as cp
import numpy as np
import networkx as nx
from datetime import datetime

# -----------------------------------------------------------------------------
# Import Evaluation Helpers from GRAN
# -----------------------------------------------------------------------------
# We attempt to import the utils. If GRAN is not in PYTHONPATH, this will fail.
try:
    from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, spectral_stats
except ImportError:
    print("\n[ERROR] Could not import 'utils.eval_helper'.")
    print("Please ensure the GRAN repository is in your PYTHONPATH.")
    print("Run: export PYTHONPATH=$PYTHONPATH:$(pwd)/../GRAN  (adjust path as needed)\n")
    sys.exit(1)

def load_graphs(file_path):
    """
    Robust graph loader that handles both:
    1. Pickled lists of graphs (how batch_train.py saves predictions)
    2. Streams of pickled individual graphs (how data_process saves splits)
    """
    graphs = []
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"Loading graphs from: {file_path}")
    with open(file_path, 'rb') as f:
        while True:
            try:
                obj = cp.load(f)
                # If the object is a list, extend; if it's a single graph, append
                if isinstance(obj, list):
                    graphs.extend(obj)
                else:
                    graphs.append(obj)
            except EOFError:
                break
            except Exception as e:
                print(f"Stopped loading due to error (likely end of stream): {e}")
                break
    
    # Filter out any non-NetworkX objects if corruption occurred, or empty graphs
    valid_graphs = [g for g in graphs if isinstance(g, (nx.Graph, nx.DiGraph))]
    print(f" -> Successfully loaded {len(valid_graphs)} graphs.")
    return valid_graphs

def print_basic_stats(name, graphs):
    if not graphs:
        print(f"{name}: No graphs.")
        return
    
    num_nodes = [g.number_of_nodes() for g in graphs]
    num_edges = [g.number_of_edges() for g in graphs]
    
    print(f"--- {name} Statistics ---")
    print(f"  Count:     {len(graphs)}")
    print(f"  Avg Nodes: {np.mean(num_nodes):.2f} (Min: {np.min(num_nodes)}, Max: {np.max(num_nodes)})")
    print(f"  Avg Edges: {np.mean(num_edges):.2f} (Min: {np.min(num_edges)}, Max: {np.max(num_edges)})")
    print("")

def main():
    parser = argparse.ArgumentParser(description='Evaluate BiGG Generated Graphs vs Ground Truth')
    parser.add_argument('-pred_file', type=str, required=True, help='Path to the generated graphs pickle file (e.g., epoch-X.graphs-Y)')
    parser.add_argument('-test_file', type=str, required=True, help='Path to the ground truth/test graphs pickle file')
    parser.add_argument('-metrics', type=str, default='degree,clustering', help='Comma separated metrics: degree,clustering,orbit,spectral')
    
    args = parser.parse_args()

    # 1. Load Graphs
    pred_graphs = load_graphs(args.pred_file)
    test_graphs = load_graphs(args.test_file)

    if len(pred_graphs) == 0 or len(test_graphs) == 0:
        print("Error: One of the graph lists is empty. Exiting.")
        sys.exit(1)

    # 2. Print Basic Stats
    print_basic_stats("Generated (Pred)", pred_graphs)
    print_basic_stats("Ground Truth (Test)", test_graphs)

    # 3. Compute Metrics
    metric_list = args.metrics.split(',')
    
    print("Computing MMD Metrics (Lower is better)...")
    print("-" * 40)

    # Degree MMD
    if 'degree' in metric_list:
        print("Computing Degree MMD...", end=' ', flush=True)
        start = datetime.now()
        mmd_degree = degree_stats(test_graphs, pred_graphs)
        print(f"Done ({datetime.now() - start})")
        print(f"Degree MMD:      {mmd_degree:.6f}")
    
    # Clustering MMD
    if 'clustering' in metric_list:
        print("Computing Clustering MMD...", end=' ', flush=True)
        start = datetime.now()
        mmd_clustering = clustering_stats(test_graphs, pred_graphs)
        print(f"Done ({datetime.now() - start})")
        print(f"Clustering MMD:  {mmd_clustering:.6f}")

    # Spectral MMD
    if 'spectral' in metric_list:
        print("Computing Spectral MMD...", end=' ', flush=True)
        start = datetime.now()
        mmd_spectral = spectral_stats(test_graphs, pred_graphs)
        print(f"Done ({datetime.now() - start})")
        print(f"Spectral MMD:    {mmd_spectral:.6f}")

    # Orbit MMD (Requires Orca)
    if 'orbit' in metric_list:
        print("Computing Orbit MMD (requires compiled 'orca' binary)...", end=' ', flush=True)
        start = datetime.now()
        try:
            mmd_orbit = orbit_stats_all(test_graphs, pred_graphs)
            print(f"Done ({datetime.now() - start})")
            print(f"Orbit MMD:       {mmd_orbit:.6f}")
        except Exception as e:
            print(f"\n[WARNING] Failed to compute Orbit MMD. Ensure 'utils/orca/orca' is compiled and accessible.")
            print(f"Error details: {e}")

    print("-" * 40)

if __name__ == '__main__':
    main()