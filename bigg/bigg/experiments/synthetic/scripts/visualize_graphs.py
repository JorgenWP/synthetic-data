import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
import os

##### Configuration #####
model_folder = "DFS-blksize--1-b-80"
graph_file_name = "epoch-25.ckpt.graphs-0"
#########################

# File paths
graph_file_path = f"../../../../results/lobster/{model_folder}/{graph_file_name}"
output_dir = "../../../../results/generated_images"

# Load graphs
print(f"Loading graphs from {graph_file_path}...")

with open(graph_file_path, 'rb') as f:
    graphs = pickle.load(f)

print(f"Successfully loaded {len(graphs)} graphs.")
 
# Create a folder to save the images
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\nDrawing graphs...")
 
# Visualize the first 5 graphs
for i, G in enumerate(graphs[:5]):
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