import networkx as nx
import matplotlib.pyplot as plt

def convert_to_networkx(custom_graph):
    # Initialize a new NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes and edges from the custom graph to the NetworkX graph
    for node_id, node in custom_graph.nodesList.items():
        G.add_node(node_id)
        for edge in node.edges:
            G.add_edge(edge.from_node.node_id, edge.to_node.node_id)
    
    return G

def visualize_graph(custom_graph):
    # Convert the custom graph to a NetworkX graph
    G = convert_to_networkx(custom_graph)
    
    # Generate positions for all nodes
    pos = nx.spring_layout(G, k=0.1, iterations=20, seed = 42)
    
    # Separate nodes by type
    min_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Min']
    max_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Max']
    
    # Draw Min nodes as circles
    nx.draw_networkx_nodes(G, pos, nodelist=min_nodes, node_color='skyblue', node_shape='o', node_size=500)
    
    # Draw Max nodes as squares
    nx.draw_networkx_nodes(G, pos, nodelist=max_nodes, node_color='lightgreen', node_shape='s', node_size=500)
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color='k', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.show()

def visualize_bipartite_graph(custom_graph):
    # Convert the custom graph to a NetworkX graph
    G = convert_to_networkx(custom_graph)
    
    # Generate positions for all nodes using the bipartite layout
    # Separate nodes by type
    nodes_set_1 = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Min']
    pos = nx.bipartite_layout(G, nodes_set_1)
    
    # Draw Min nodes as circles
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_set_1, node_color='skyblue', node_shape='o', node_size=500)
    
    # Draw Max nodes as squares
    nodes_set_2 = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Max']
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_set_2, node_color='lightgreen', node_shape='s', node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='k', arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.show()

def visualize_winners(custom_graph, minlist, maxlist):
    G = convert_to_networkx(custom_graph)
    pos = nx.spring_layout(G, k=0.1, iterations=20, seed=42)

    # Initial draw for Min and Max nodes in grey to set default colors and shapes
    min_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Min']
    max_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Max']
    nx.draw_networkx_nodes(G, pos, nodelist=min_nodes, node_color='lightgrey', node_shape='o', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=max_nodes, node_color='lightgrey', node_shape='s', node_size=500)

    # Highlight winner Min nodes in blue
    winner_min_nodes = [node for node in min_nodes if node in minlist]
    if winner_min_nodes:  # Check if list is not empty
        nx.draw_networkx_nodes(G, pos, nodelist=winner_min_nodes, node_color='blue', node_shape='o', node_size=500)
    
    # Highlight winner Max nodes in red
    winner_max_nodes = [node for node in max_nodes if node in maxlist]
    if winner_max_nodes:  # Check if list is not empty
        nx.draw_networkx_nodes(G, pos, nodelist=winner_max_nodes, node_color='red', node_shape='s', node_size=500)

    # Draw nodes in minlist but are Max type, and vice versa, based on your additional criteria
    min_nodes_as_max_winner = [node for node in min_nodes if node in maxlist]
    max_nodes_as_min_winner = [node for node in max_nodes if node in minlist]
    if min_nodes_as_max_winner:
        nx.draw_networkx_nodes(G, pos, nodelist=min_nodes_as_max_winner, node_color='red', node_shape='o', node_size=500)
    if max_nodes_as_min_winner:
        nx.draw_networkx_nodes(G, pos, nodelist=max_nodes_as_min_winner, node_color='blue', node_shape='s', node_size=500)

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color='k', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.show()

def visualize_bipartite_winners(custom_graph, minlist, maxlist):
    G = convert_to_networkx(custom_graph)
    
    # Generate positions for all nodes using the bipartite layout
    # Assuming min_nodes are one set and max_nodes are another set in the bipartite graph
    min_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Min']
    pos = nx.bipartite_layout(G, min_nodes)

    # Initial draw for Min and Max nodes in grey to set default colors and shapes
    max_nodes = [node_id for node_id, node in custom_graph.nodesList.items() if node.node_type == 'Max']
    nx.draw_networkx_nodes(G, pos, nodelist=min_nodes, node_color='lightgrey', node_shape='o', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=max_nodes, node_color='lightgrey', node_shape='s', node_size=500)

    # Highlight winner Min nodes in blue and Max nodes in red
    # Note that the shape of the node (circle/square) is determined by the node_type
    for node_id in min_nodes:
        color = 'blue' if node_id in minlist else 'red' if node_id in maxlist else 'lightgrey'
        nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_color=color, node_shape='o', node_size=500)
    for node_id in max_nodes:
        color = 'blue' if node_id in minlist else 'red' if node_id in maxlist else 'lightgrey'
        nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_color=color, node_shape='s', node_size=500)

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color='k', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.show()