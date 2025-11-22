import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
import os

##### Configuration #####
data_dir = "results" # 'data' or 'results'
file_path = f"lobster/DFS-blksize--1-b-80/epoch-25.ckpt.graphs-0"
# file_path = "Transactions/transactions-BFS/train-graphs.pkl"
#########################

# File paths
graph_file_path = f"../../../../{data_dir}/{file_path}"
output_dir = "../../../../results/generated_images"

# Load graphs
print(f"Loading graphs from {graph_file_path}...")

graphs = []
with open(graph_file_path, 'rb') as f:
    if data_dir == "results":
        graphs = pickle.load(f)
    elif data_dir == "data":
        try:
            while True:
                graphs.append(pickle.load(f))
        except EOFError:
            pass  # Finished reading all graphs
    else:
        raise ValueError(f"Unknown data_dir: {data_dir}")

print(f"Successfully loaded {len(graphs)} graph(s).")
 
# Create a folder to save the images
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\nDrawing graphs...")
 
# Visualize the first 5 graphs
num_to_draw = min(5, len(graphs))

for i, G in enumerate(graphs[:num_to_draw]):
    print(f'Drawing graph {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
   
    plt.figure(figsize=(8, 8))
   
    # We use a spring layout which tries to position nodes naturally
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw(G, pos,
            node_size=50,
            node_color='blue',
            edge_color='gray',
            with_labels=False,
            alpha=0.7)
    
    plt.suptitle(f"Graph {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save to file
    save_file_name = os.path.join(output_dir, f"graph_{i}.png")
    plt.savefig(save_file_name, bbox_inches='tight')
    plt.close()
   
    print(f"Saved graph image to {save_file_name}")
 
print(f"\nDone! Check the 'results/generated_images' folder.")