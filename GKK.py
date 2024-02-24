import os
from bipartite import Graph as Bipartite, Node as BNode
class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type  # 'Min' or 'Max'
        self.edges = []  # List of edges connected to this node
        self.totalPotential = 0

    def add_edge(self, edge):
        self.edges.append(edge)

    def __repr__(self):
        return f"Node({self.node_id}, Type: {self.node_type})"
    def printEdges(self):
        for edge in self.edges:
            print(edge)


class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    def __repr__(self):
        return f"Edge(from: {self.from_node.node_id}, to: {self.to_node.node_id}, weight: {self.weight})"


class Graph:
    def __init__(self):
        self.nodesList = {}  # Dictionary to store nodes by their ID
        self.nodes = []
        self.edges = []  # List to store all edges
        self.potentials = {}
        self.memo = {}
        self.trivialsMin = []
        self.trivialsMax = []
    def add_node(self, node):
        """Add a node to the graph."""
        if node.node_id not in self.nodesList:
            self.nodesList[node.node_id] = node
            self.nodes.append(node)
            self.potentials[node.node_id] = 0
    def add_edge(self, from_node_id, to_node_id, weight):
        """Add an edge to the graph."""
        if from_node_id in self.nodesList and to_node_id in self.nodesList:
            from_node = self.nodesList[from_node_id]
            to_node = self.nodesList[to_node_id]
            edge = Edge(from_node, to_node, weight)
            self.edges.append(edge)
            from_node.add_edge(edge)
    def set_potential(self, node, potential_value):
        """Set the potential for a given node."""
        if node.node_id in self.potentials:
            self.potentials[node.node_id] = potential_value

    def get_potential(self, node):
        """Get the potential for a given node."""
        return self.potentials.get(node.node_id, None)
    def reset_potentials(self):
        for node in self.nodes:
            node.totalPotential = node.totalPotential + self.potentials[node.node_id]
            self.set_potential(node, 0)
        return
    def apply_modified(self, modified_weights):
        for edge in self.edges:
            edge.weight = modified_weights[edge]
        return
    def __repr__(self):
        return f"Graph with {len(self.nodes)}, nodes and {len(self.edges)} edges"
    
    def calculate_modified_weights(self):
        """Calculate modified weights for all edges based on current potentials."""
        modified_weights = {}
        for edge in self.edges:
            # Adjust the weight of the edge using the potentials of its nodes
            adjusted_weight = edge.weight + self.potentials[edge.to_node.node_id] - self.potentials[edge.from_node.node_id]
            # Store the modified weight with a tuple of the edge's start and end node IDs
            modified_weights[edge] = adjusted_weight
        return modified_weights

    def classify_nodes(self,nodes, modified_weights):
        """ this function is designed to find our partition of N and P in the graph """
        """ To do so we need to satisfy the following rules:"""
        """ For N* (in mins favour) :
            if node is min:
                - there exists a single negative outgoing edge
                OR
                - if there exists a 0 edge then by taking one of these edges we will eventually be able to 
                  take a negative edge, i.e. is able to reach another N* node under optimal play"""
        """ if node is max:
                - all outgoing edges are either negative or zero in which following that path under optimal play will reach a negative edge, i.e. """
        """     ends up only reaching nodes in N* """

        """ For P* (in max favour):
            if node is min:
                - all outgoing edges are either positive or zero in which following that path under optimal play will reach a positive edge, i.e. """
        """     ends up only reaching nodes in P*"""
        """ if node is max:
                - there exists a single positive edge
                OR
                -  if there exists a 0 edge then by taking one of these edges we will eventually be able to 
                  take a positive edge, i.e. is able to reach another P* node under optimal play"""
        N_star = set()
        P_star = set()
        self.memo = {}

        for node in nodes:
            self.DPeval(node, modified_weights)
        for node in nodes:
            if (self.memo[node.node_id] is True):
                N_star.add(node)
            else:
                P_star.add(node)
        return P_star, N_star
    
    # Helper function to create P* and N* in linear time through use of Dynamic programming
    def DPeval(self,node, modified_weights):
        isN = False
        if node.node_id in self.memo:
            return self.memo[node.node_id]
        else:
            if node.node_type == "Min":
                for edge in node.edges:
                    if (modified_weights[edge] < 0):
                        isN = True
                        break
                    elif (modified_weights[edge] == 0):
                        isN = isN or self.DPeval(edge.to_node, modified_weights)
                    else:
                        pass
            else:
                # node is Max so check if its in P* else it is in N*
                isN = True
                for edge in node.edges:
                    if (modified_weights[edge] > 0):
                        isN = False
                        break
                    elif(modified_weights[edge] == 0):
                        isN = isN and self.DPeval(edge.to_node, modified_weights)
                    else:
                        pass
            self.memo[node.node_id] = isN
            return self.memo[node.node_id]


    def calculateDeltaPlusMin(self, N_star, P_star):
        # Initialize delta_min to a large number to ensure it gets updated
        delta_min = float('inf')
        
        # Iterate over all edges in the graph
        for edge in self.edges:
            # Check if the from_node is in P_star and is a Min node, and to_node is in N_star
            if edge.from_node in P_star and edge.from_node.node_type == 'Min' and edge.to_node in N_star:
                # Update delta_min with the minimum weight found
                delta_min = min(delta_min, edge.weight)

        # return the result inf if no edges
        if delta_min <= 0:
            return float('inf')
        else:
            return delta_min 

    def calculateDeltaMinusMin(self, N_star, P_star):
        SN = self.get_SN(N_star)
        delta_plus_max_values = []
        for node in SN:
            min_negative_edge_weight = float('inf')
            for edge in node.edges:
                min_negative_edge_weight = min(min_negative_edge_weight, edge.weight)
            if min_negative_edge_weight < float('inf'):
                delta_plus_max_values.append(min_negative_edge_weight)
                min_negative_edge_weight = float('inf')
        if delta_plus_max_values:
            return max(delta_plus_max_values) if max(delta_plus_max_values) < 0 else -float('inf')
        else:
            return -float('inf')
    def get_SN(self, N_star):
        SN = set()

        # Iterate over all nodes in V_Max that are also in P_star
        for node in N_star :
            fax = True
            if node.node_type == "Min":
                # Check all incoming edges from nodes in P_star
                for edge in node.edges:
                    if edge.to_node in N_star:
                        fax = (edge.weight > 0) and fax
                    else:
                        pass
                if fax:
                    SN.add(node)
            else:
                pass

        return SN

    def calculateDeltaPlusMax(self, N_star, P_star):
        SP = self.get_SP(P_star)
        delta_plus_max_values = []
        for node in SP:
            max_positive_edge_weight = -float('inf')

            for edge in node.edges:
                max_positive_edge_weight = max(max_positive_edge_weight, edge.weight)
            if max_positive_edge_weight > -float('inf'):
                delta_plus_max_values.append(max_positive_edge_weight)
                max_positive_edge_weight = -float('inf')
        if delta_plus_max_values:
            return min(delta_plus_max_values) if min(delta_plus_max_values) > 0 else float('inf')
        else:
            return float('inf')
        
    
    def get_SP(self, P_star):
        SP = set()

        # Iterate over all nodes in V_Max that are also in P_star
        for node in P_star :
            fax = True
            if node.node_type == "Max":
                # Check all incoming edges from nodes in P_star
                for edge in node.edges:
                    if edge.to_node in P_star:
                        fax = (edge.weight < 0) and fax
                    else:
                        pass
                if fax:
                    SP.add(node)
            else:
                pass

        return SP

    def calculateDeltaMinusMax(self, N_star, P_star):
        # Initialize delta_min to a large number to ensure it gets updated
        delta_max = -float('inf')
        
        # Iterate over all edges in the graph
        for edge in self.edges:
            # Check if the from_node is in P_star and is a Min node, and to_node is in N_star
            if edge.from_node in N_star and edge.from_node.node_type == 'Max' and edge.to_node in P_star:
                # Update delta_min with the minimum weight found
                delta_max = max(delta_max, edge.weight)

        # return the result inf if no edges
        if delta_max >= 0:
            return -float('inf')
        else:
            return delta_max 
    def calculate_Delta(self, N_star, P_star):
        minus = max(self.calculateDeltaMinusMax(N_star, P_star), self.calculateDeltaMinusMin(N_star,P_star) )
        plus = min(self.calculateDeltaPlusMin(N_star,P_star), self.calculateDeltaPlusMax(N_star,P_star))
        if minus > 0:
            minus = -float('inf')
        if plus < 0:
            plus = float('inf')
        return min(-minus,plus)

    def extract_strategies(self, filename):
        N_star, P_star, iterations = self.run_gkk_algorithm()
        newfile = os.path.join('GKK', os.path.basename(filename))
        with open(newfile, 'w') as file:
            file.write("Iteration Count: " + str(iterations) + "\n")
            all_node_ids = sorted(set(self.trivialsMin + self.trivialsMax + [node.node_id for node in self.nodes]))
            energy_value = 0
            for node_id in all_node_ids:
                if node_id in self.trivialsMin:
                    energy_value = 0  # Assuming trivialMin nodes have an energy value of 0
                elif node_id in self.trivialsMax:
                    energy_value = float('inf')  # Assuming trivialMax nodes have an energy value of inf
                else:
                    node = self.nodesList[node_id]
                    if node in P_star:
                        energy_value = float('inf')
                        print(str(node.node_id) + " , has a energy value of: Infinity") 
                    else:
                        energy_value = 0
                        print(str(node.node_id) + " , has a energy value of: " + str(node.totalPotential))
                whoWins = ''
                if energy_value != float('inf'):
                    whoWins = 'Min'
                else:
                    whoWins = 'Max'
                file.write(f"{node_id} Wins for: {whoWins}\n")
                print(f"Node {node_id} Wins for: {whoWins}")
        return N_star,P_star

    def run_gkk_algorithm(self):
        iterations = 0
        P_star = set()
        N_star = set()
        delta = 0
        counter = 0
        if not self.nodes:
            return N_star,P_star, iterations
        while delta != float('inf'):
            self.reset_potentials()
            for node in P_star:
                self.set_potential(node, delta)
            modified = self.calculate_modified_weights()
            P_star, N_star = self.classify_nodes(self.nodes, modified)
            self.apply_modified(modified)
            delta = self.calculate_Delta(N_star, P_star)
            iterations = iterations + 1
        print("num of iterations is: ", iterations)
        self.reset_potentials()
        return N_star,P_star, iterations

    # Additional methods specific to the GKK algorithm can be added here.


def createGraph(filename, useBipartite):
    print(filename)
    if useBipartite:
        graph = Bipartite()
    else:
        graph = Graph()
    with open(filename, 'r') as file:
        linenumber = 0
        for line in file:
            line = line.strip()
            if linenumber == 0:
                if line:
                    graph.trivialsMin = list(map(int, line.split(',')))
            elif linenumber == 1:
                if line:
                    graph.trivialsMax = list(map(int, line.split(',')))
            else: 
                parts = line.strip().split(' ')
                identifier = int(parts[0])
                type = int(parts[1])
                successors = parts[2].split(',')
                successors = list(map(int, successors))
                weights = parts[3].split(',')
                weights = list(map(int, weights))
                if useBipartite:
                    if type == 0:
                        graph.add_node(BNode(identifier, 'Min'))
                    else:
                        graph.add_node(BNode(identifier, 'Max'))
                else:
                    if type == 0:
                        graph.add_node(Node(identifier, 'Min'))
                    else:
                        graph.add_node(Node(identifier, 'Max'))
            linenumber += 1
    with open(filename, 'r') as file:
        length = linenumber - 2
        linenumber = 0
        for line in file:
            if line.strip() and linenumber > 1:  # Check if line is not empty
                parts = line.strip().split(' ')
                identifier = int(parts[0])
                type = int(parts[1])
                successors = parts[2].split(',')
                successors = list(map(int, successors))
                weights = parts[3].split(',')
                weights = list(map(int, weights))
                for successor , weight in zip(successors,weights):
                    graph.add_edge(identifier, successor, (-length) ** weight)
            linenumber += 1
    if useBipartite:
        newGraph = Graph()
        graph.convertToBipartite()
        newGraph.trivialsMin = sorted([node.node_id for node, winner in graph.trivials.items() if winner == 'Min'])
        newGraph.trivialsMax = sorted([node.node_id for node, winner in graph.trivials.items() if winner == 'Max'])
        for node in graph.nodes:
            newGraph.add_node(Node(node.node_id, node.node_type))
        for edge in graph.edges:
            newGraph.add_edge(edge.from_node.node_id, edge.to_node.node_id, edge.weight)
        return newGraph
    else:
        return graph


# Example Usage
# Add nodes to the graph. The type ('Min' or 'Max') for each node is assumed based on the image.
# Squares are 'Max' nodes and circles are 'Min' nodes as per mean-payoff game conventions.
# graph = Graph()
# one = Node(1, 'Max')
# two = Node(2, 'Min')
# three = Node(3, 'Max')
# foure = Node(4, 'Min')
# five = Node(5, 'Max')
# six = Node(6, 'Min')
# graph.add_node(one)  # Assuming square nodes are 'Max'
# graph.add_node(two)
# graph.add_node(three)  # Assuming circle nodes are 'Min'
# graph.add_node(foure)
# graph.add_node(five)
# graph.add_node(six)

# # Add edges to the graph. The weight for each edge is taken from the image.
# # Note that the image has nodes 'u' and 'w' as Max nodes (squares), and nodes 'v' and 'x' as Min nodes (circles).
# graph.add_edge(1, 2, -8)
# graph.add_edge(2, 1, 4)
# graph.add_edge(2, 4, 2)
# graph.add_edge(2, 3, -2)
# graph.add_edge(3, 2, 7)
# graph.add_edge(3, 4, 1)
# graph.add_edge(4, 3, -2)
# graph.add_edge(4, 5, -5)
# graph.add_edge(5, 3, 3)
# graph.add_edge(5, 6, 1)
# graph.add_edge(6, 5, 2)
# graph.add_edge(6, 6, 1)
# print(graph)
# print(one)
# one.printEdges()
# print(two)
# two.printEdges()
# print(three)
# three.printEdges()
# print(foure)
# foure.printEdges()
# print(five)
# five.printEdges()
# print(six)
# six.printEdges()
directory = 'PVIsafeOinkEGs'
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    bipartite = True
    if os.path.isfile(file_path):
        graph = createGraph(file_path, bipartite)
        print(graph)
        for node in graph.nodes:
            print(node)
            node.printEdges()
        if bipartite:
            file_path = os.path.join(directory, 'b' + filename)
        N_star, P_star = graph.extract_strategies(file_path)
        print(N_star)
        print(P_star)