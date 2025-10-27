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
print("Generating graphs for the demonstration...")
regular_graph, _ = watts_strogatz_graph(N, K, 0)
small_world_graph, initial_edges = watts_strogatz_graph(N, K, P_small)
random_graph, _ = watts_strogatz_graph(N, K, 1)
print("Graphs generated successfully.")

# --- Visualization Sequence ---

# 1. Show the Regular Graph (p=0)
pos = nx.circular_layout(regular_graph)
plt.figure(figsize=(8, 8))
plt.title("1. Regular Network (p=0)", fontsize=16)
nx.draw(regular_graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
print("\nShowing the Regular Network. CLOSE THE GRAPH WINDOW to proceed to the next step.")
plt.show()


# 2. Show the Small-World Graph (The "Sweet Spot")
plt.figure(figsize=(8, 8))
plt.title(f"2. The 'Sweet Spot' (p={P_small})", fontsize=16)
# Identify which edges are original and which are rewired shortcuts
original_edges = [edge for edge in small_world_graph.edges() if edge in initial_edges or (edge[1], edge[0]) in initial_edges]
rewired_edges = [edge for edge in small_world_graph.edges() if edge not in original_edges and (edge[1], edge[0]) not in original_edges]
# Draw the components
nx.draw_networkx_nodes(small_world_graph, pos, node_color='skyblue', node_size=700)
nx.draw_networkx_labels(small_world_graph, pos)
nx.draw_networkx_edges(small_world_graph, pos, edgelist=original_edges, edge_color='gray', width=1.5)
nx.draw_networkx_edges(small_world_graph, pos, edgelist=rewired_edges, edge_color='red', width=2.0, style='dashed')

# Create a custom legend to ensure it displays correctly
red_dashed_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label='Rewired Shortcuts')
plt.legend(handles=[red_dashed_line], loc='upper right')
print("\nShowing the Small-World Network. Note the red 'shortcuts'. This is the 'Sweet Spot'.")
print("CLOSE THE GRAPH WINDOW to see the final graph.")
plt.show()


# 3. Show the Random Graph (p=1)
plt.figure(figsize=(8, 8))
plt.title("3. Random Network (p=1)", fontsize=16)
nx.draw(random_graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
print("\nShowing the Random Network. This is the end of the demonstration.")
plt.show()

