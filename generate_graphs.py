import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random

def watts_strogatz_graph(n, k, p):
    """
    Generates a Watts-Strogatz small-world graph.

    Args:
        n (int): The number of nodes.
        k (int): Each node is joined with its `k` nearest neighbors in a ring topology.
                 `k` must be an even integer.
        p (float): The probability of rewiring each edge.

    Returns:
        networkx.Graph: The generated small-world graph.
    """
    if k % 2 != 0:
        raise ValueError("k must be an even integer.")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")

    # 1. START: Create a regular ring lattice
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)

    # Connect each node to its k nearest neighbors
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            G.add_edge(i, neighbor)
            
    initial_edges = set(G.edges())

    # 2. EVOLVE: Rewire edges with probability p
    for u, v in list(initial_edges):
        if random.random() < p:
            # Choose a new node to connect to, different from u and its neighbors
            possible_new_nodes = set(nodes) - {u} - set(nx.neighbors(G, u))
            if not possible_new_nodes:
                continue

            new_v = random.choice(list(possible_new_nodes))
            
            # Rewire the edge
            G.remove_edge(u, v)
            G.add_edge(u, new_v)

    return G, initial_edges

# --- Parameters for the model ---
N = 20  # Number of nodes
K = 4   # Number of nearest neighbors for the initial ring
P_small = 0.2 # Small rewiring probability for the "sweet spot"

# --- Generate all three graphs for the presentation ---
print("Generating graphs for the website...")
regular_graph, _ = watts_strogatz_graph(N, K, 0)
small_world_graph, initial_edges = watts_strogatz_graph(N, K, P_small)
random_graph, _ = watts_strogatz_graph(N, K, 1)
print("Graphs generated successfully.")

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import os

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), 'website', 'images')
os.makedirs(output_dir, exist_ok=True)

# --- Save visualizations as images ---
pos = nx.circular_layout(regular_graph)

# 1. Regular Graph (p=0)
plt.figure(figsize=(10, 10))
plt.title("Regular Network (p=0)", fontsize=18, fontweight='bold', pad=20)
nx.draw(regular_graph, pos, with_labels=True, node_color='#4A90E2', node_size=800, 
        edge_color='#7C8B9E', width=2.0, font_size=14, font_weight='bold', font_color='white')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regular_network.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved regular_network.png")
plt.close()

# 2. Small-World Graph (The "Sweet Spot")
plt.figure(figsize=(10, 10))
plt.title(f"Small-World Network - The 'Sweet Spot' (p={P_small})", fontsize=18, fontweight='bold', pad=20)
# Identify which edges are original and which are rewired shortcuts
original_edges = [edge for edge in small_world_graph.edges() if edge in initial_edges or (edge[1], edge[0]) in initial_edges]
rewired_edges = [edge for edge in small_world_graph.edges() if edge not in original_edges and (edge[1], edge[0]) not in original_edges]
# Draw the components
nx.draw_networkx_nodes(small_world_graph, pos, node_color='#4A90E2', node_size=800)
nx.draw_networkx_labels(small_world_graph, pos, font_size=14, font_weight='bold', font_color='white')
nx.draw_networkx_edges(small_world_graph, pos, edgelist=original_edges, edge_color='#7C8B9E', width=2.0)
nx.draw_networkx_edges(small_world_graph, pos, edgelist=rewired_edges, edge_color='#E74C3C', width=3.0, style='dashed')

# Create a custom legend
red_dashed_line = mlines.Line2D([], [], color='#E74C3C', linestyle='--', linewidth=3, label='Rewired Shortcuts')
gray_solid_line = mlines.Line2D([], [], color='#7C8B9E', linestyle='-', linewidth=2, label='Original Edges')
plt.legend(handles=[gray_solid_line, red_dashed_line], loc='upper right', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'small_world_network.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved small_world_network.png")
plt.close()

# 3. Random Graph (p=1)
plt.figure(figsize=(10, 10))
plt.title("Random Network (p=1)", fontsize=18, fontweight='bold', pad=20)
nx.draw(random_graph, pos, with_labels=True, node_color='#4A90E2', node_size=800, 
        edge_color='#7C8B9E', width=2.0, font_size=14, font_weight='bold', font_color='white')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'random_network.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved random_network.png")
plt.close()

print("\n✅ All graph images generated successfully in website/images/")
