import pickle
import networkx as nx
import pandas as pd
import os
import sys

source_dir = "../../data/Transactions/transactions-BFS"

output_dir = "../../results/generated_csv/validation_graphs"

target_files = [
    "test-graphs.pkl", 
    "val-graphs.pkl"
]

# ---------------------

def save_graph_to_csv(G, filename, folder):
    """Saves a single networkx graph to a CSV file with Source/Target headers."""
    full_path = os.path.join(folder, filename)
    
    try:
        
        df = nx.to_pandas_edgelist(G)
        
        df = df.rename(columns={'source': 'Source', 'target': 'Target'})
        
        # Save to CSV
        df.to_csv(full_path, index=False)
        return True
    except Exception as e:
        print(f"  [!] Failed to save {filename}: {e}")
        return False

def process_file(filename):
    file_path = os.path.join(source_dir, filename)
    file_tag = filename.replace("-graphs.pkl", "")
    
    print(f"\nProcessing {filename}...")
    
    if not os.path.exists(file_path):
        print(f"  [Error] File not found: {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        
        if isinstance(data, list):
            graphs = data
        else:
            graphs = [data]
            
        print(f"  Found {len(graphs)} graphs.")
        
        count = 0
        for i, G in enumerate(graphs):

            csv_name = f"{file_tag}_graph_{i}.csv"
            
            if save_graph_to_csv(G, csv_name, output_dir):
                count += 1
                
        print(f"  Successfully saved {count} CSV files.")

    except Exception as e:
        print(f"  [Error] Failed to load pickle: {e}")

def main():
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Output directory exists: {output_dir}")

    for pkl_file in target_files:
        process_file(pkl_file)

    print("\nAll operations complete.")

if __name__ == "__main__":
    main()