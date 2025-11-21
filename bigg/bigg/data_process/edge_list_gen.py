import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
import os
 
### Configuration ###
file_path = "../../data/Transactions/transactions-BFS/train-graphs.pkl"
#####################

output_dir = "../../results/generated_csv"

print(f"Loading graphs from {file_path}...")

graphs = []
with open(file_path, 'rb') as f:
    while True:
        try:
            g = pickle.load(f)
            graphs.append(g)
        except EOFError:
            break
 
print(f"Successfully loaded {len(graphs)} graphs.")
 
# Create a folder to save csv files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\nSaving graphs as CSV files in {output_dir}...")

for i, G in enumerate(graphs[:1]):
    nx.write_edgelist(G, os.path.join(output_dir, f"graph_{i}.csv"), data=False, delimiter=",")
    print(f"Saved graph_{i}.csv")
 
print("\nDone! Check the 'results/generated_csv' folder.")