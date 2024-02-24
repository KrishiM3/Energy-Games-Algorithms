from scipy.optimize import linprog
import numpy as np
import os
import sys
from bipartite import Graph as Bipartite, Node as BNode
class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type  # 'Min' or 'Max'
        self.edges = []  # List of edges connected to this node
        # self.totalPotential = 0

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
        self.nodesID = []
        self.nodes = []
        self.edges = []  # List to store all edges
        # self.potentials = {}
        self.x = {} # signifies our PolPoten
        self.memo = {}
        self.trivialsMin = []
        self.trivialsMax = []
    def add_node(self, node):
        """Add a node to the graph."""
        if node.node_id not in self.nodesList:
            self.nodesList[node.node_id] = node
            self.nodesID.append(node.node_id)
            self.nodes.append(node)
            self.x[node.node_id] = 0
            # self.potentials[node.node_id] = 0
    def add_edge(self, from_node_id, to_node_id, weight):
        """Add an edge to the graph."""
        if from_node_id in self.nodesList and to_node_id in self.nodesList:
            from_node = self.nodesList[from_node_id]
            to_node = self.nodesList[to_node_id]
            edge = Edge(from_node, to_node, weight)
            self.edges.append(edge)
            from_node.add_edge(edge)

    def initX(self):
        """We want a preliminary polpoten so we must find any random point which satisfies our inequalities"""
        c = [1] * len(self.x)  # Coefficients for x are 0, coefficient for ε is -1 (since linprog minimizes)

        # Constraints
        A = []
        b_vec = []
        for edge in self.edges:
            a, b = edge.from_node.node_id, edge.to_node.node_id
            row = [0] * len(self.x)  # Initialize with coefficients for x (0) 

            if self.nodesList[a].node_type == 'Max':
                row[list(self.x.keys()).index(a)] = -1 
                row[list(self.x.keys()).index(b)] = 1
                A.append(row)  # For Max nodes, xa + ε * x^+_a - xb - ε * x^+_b >= w(e)
                b_vec.append(-edge.weight)
            else:
                row[list(self.x.keys()).index(a)] = 1 
                row[list(self.x.keys()).index(b)] = -1
                A.append(row)
                b_vec.append(edge.weight)

        # Bounds
        bounds = [(0, None)] * len(self.x)  # No bounds on x, ε >= 0

        # Solve the linear program
        result = linprog(c, A_ub=A, b_ub=b_vec, bounds=bounds, method='simplex')
        node_to_x_value = {node_id: x_value for node_id, x_value in zip(self.x.keys(), result.x)}
        self.x = node_to_x_value
        return 
    
    def stronglyViolatingCheck(self, delta):
        """ Returns True if there exists an edge which is strongly violating"""
        for edge in self.edges:
            if edge.from_node.node_type == "Min":
                if delta[edge.from_node.node_id] > 0 and delta[edge.to_node.node_id] < 0:
                    return True
                continue
            else:
                if delta[edge.from_node.node_id] < 0 and delta[edge.to_node.node_id] > 0:
                    return True
                continue
        return False
    
    def obtainTightEdges(self, shift):
        tightEdges = set()
        if shift is None:
            for edge in self.edges:
                if self.x[edge.from_node.node_id] == self.x[edge.to_node.node_id] + edge.weight:
                    tightEdges.add(edge)
            pass
        else:
            for edge in self.edges:
                if self.x[edge.from_node.node_id] + shift[edge.from_node.node_id] == self.x[edge.to_node.node_id] + shift[edge.to_node.node_id] + edge.weight:
                    tightEdges.add(edge)
            pass
        return tightEdges

    def calculate_Delta(self, edges):
        """ Obtains the feasible shift vector for our feasible shift w.r.t our current x in PolPoten (DNPG's)"""
        """Instead of calculating all of the delta values we only need to calculate the +ve ones, As our feasible shift
            is the characterisation vector of these."""
        # delta = {}
        # adjacencyCount = {}
        # adjacencyList = {}
        # for node in self.nodes:
        #     delta[node.node_id] = 0
        #     adjacencyCount[node.node_id] = 0
        #     adjacencyList[node.node_id] = []

        # for edge in edges:
        #     adjacencyCount[edge.from_node.node_id] = adjacencyCount[edge.from_node.node_id] + 1
        #     adjacencyList[edge.from_node.node_id].append(edge)
        # for key in adjacencyCount:
        #     if adjacencyCount[key] == 0 and key.node_type == "Min":
        #         delta[key] = 1
        #     elif adjacencyCount[key] == 0 and key.node_type == "Max":
        #         delta[key] = -1
        
        # for i in range(len(adjacencyCount)):
        #     for node in self.nodes:
        #         if delta[node.node_id] != 0:
        #             continue
        #         else:
        #             if node.node_type == "Max":
        #                 maxDelta = -1
        #                 for edge in adjacencyList[node.node_id]:
        #                     maxDelta = max(maxDelta, delta[edge.to_node.node_id] * 1/2)
        #                 delta[node.node_id] = maxDelta
        #             else:
        #                 minDelta = 1
        #                 for edge in adjacencyList[node.node_id]:
        #                     minDelta = min(minDelta, delta[edge.to_node.node_id] * 1/2)
        #                 delta[node.node_id] = minDelta 
        delta = {}
        adjacencyCount = {}
        adjacencyList = {}
        for node in self.nodes:
            delta[node.node_id] = 0
            adjacencyCount[node.node_id] = 0
            adjacencyList[node.node_id] = []

        for edge in edges:
            adjacencyCount[edge.from_node.node_id] += 1
            adjacencyList[edge.from_node.node_id].append(edge)

        for node in self.nodes:
            node_id = node.node_id
            # Check if the node is a sink (no outgoing edges)
            if adjacencyCount[node_id] == 0:
                # Assign delta according to whether the node is from V_Min or V_Max
                delta[node_id] = 1 if node.node_type == "Min" else -1

        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                current_delta = delta[node.node_id]
                if node.node_type == "Max":
                    maxDelta = -1
                    for edge in adjacencyList[node.node_id]:
                        maxDelta = max(maxDelta, delta[edge.to_node.node_id] * 1/2)
                    delta[node.node_id] = maxDelta
                else:
                    minDelta = 1
                    for edge in adjacencyList[node.node_id]:
                        minDelta = min(minDelta, delta[edge.to_node.node_id] * 1/2)
                    delta[node.node_id] = minDelta
                if current_delta != delta[node.node_id]:
                    changed = True

        return delta

    def calculate_characterisation(self, delta):
        """Obtains X^+"""
        xPlus = {}
        for node in delta:
            if delta[node] > 0:
                xPlus[node] = 1
            else:
                xPlus[node] = 0
        return xPlus
    def compute_EpsilonMax(self , xPlus):
        """ Computes Episilion Max using Simplex Method"""
        c = [-1]  # Minimize -ε to maximize ε

        A = []
        B = []
        for edge in self.edges:
            a = edge.from_node
            b = edge.to_node
            weight = edge.weight
            epsilon_coef = xPlus[a.node_id] - xPlus[b.node_id]
            constant_term = weight + self.x[b.node_id] - self.x[a.node_id]

            # Adjust inequality based on node type
            if a.node_type == 'Max':
                # For Max nodes, ε * (x^+_a - x^+_b) >= w(e) + x_b - x_a
                # Negate to fit into the '≤' form for linprog
                row = [-epsilon_coef]
                A.append(row)
                B.append(-constant_term)
            else:
                # For Min nodes, ε * (x^+_a - x^+_b) <= w(e) + x_b - x_a
                row = [epsilon_coef]
                A.append(row)
                B.append(constant_term)

        bounds = [(0, None)]  # ε >= 0

        result = linprog(c, A_ub=A, b_ub=B, bounds=bounds, method='simplex')

        max_epsilon = -result.fun  # Negate the minimized value to get the maximum ε
        return max_epsilon
    
    def updateShift(self, epsilon, xPlus):
        for node in xPlus:
            xPlus[node] = xPlus[node] * epsilon
        return xPlus

    def realiseGraph(self, edges):
        """ Obtains a new x in PolPoten using the tight edges in x + epsilion * X^+ and changing the inequalities into 
            equalities
        """
        c = [0] * len(self.x)  # Coefficients for x are 0, coefficient for ε is -1 (since linprog minimizes)

        # Constraints
        A = []
        b_vec = []
        for edge in self.edges:
            a, b = edge.from_node.node_id, edge.to_node.node_id 
            if edge in edges:
                row = [0] * len(self.x)  # Initialize with coefficients for x (0)
                row[list(self.x.keys()).index(a)] = -1 
                row[list(self.x.keys()).index(b)] = 1
                A.append(row)  # For Max nodes, xa + ε * x^+_a - xb - ε * x^+_b >= w(e)
                b_vec.append(-edge.weight)
                row = [0] * len(self.x)  # Initialize with coefficients for x (0)
                row[list(self.x.keys()).index(a)] = 1 
                row[list(self.x.keys()).index(b)] = -1
                A.append(row)
                b_vec.append(edge.weight)
            else:
                row = [0] * len(self.x)
                if self.nodesList[a].node_type == 'Max':
                    row[list(self.x.keys()).index(a)] = -1 
                    row[list(self.x.keys()).index(b)] = 1
                    A.append(row)  # For Max nodes, xa + ε * x^+_a - xb - ε * x^+_b >= w(e)
                    b_vec.append(-edge.weight)
                else:
                    row[list(self.x.keys()).index(a)] = 1 
                    row[list(self.x.keys()).index(b)] = -1
                    A.append(row)
                    b_vec.append(edge.weight)

        # Bounds
        bounds = [(0, None)] * len(self.x)  # No bounds on x, ε >= 0

        # Solve the linear program
        # Solve the linear program with increased verbosity
        options = {'disp': True}
        result = linprog(c, A_ub=A, b_ub=b_vec, bounds=bounds, method='simplex', options=options)

        if not result.success:
            print("Failure: ", result.message)  # Provides the reason for failure
            sys.exit()
        node_to_x_value = {node_id: x_value for node_id, x_value in zip(self.x.keys(), result.x)}
        return node_to_x_value
 
        # # Initialize matrices A and b
        # A = [[0 for _ in range(len(self.nodes))] for _ in range(len(edges))]
        # b = [0 for _ in range(len(edges))]

        # # Fill in the matrices based on the edges
        # for i, edge in enumerate(edges):
        #     from_node = edge.from_node
        #     to_node = edge.to_node
        #     if from_node.node_type == "Max":
        #         A[i][from_node.node_id - 1] = -1
        #         A[i][to_node.node_id - 1] = 1
        #         b[i] = -edge.weight
        #     else:

        #         A[i][from_node.node_id - 1] = 1
        #         A[i][to_node.node_id - 1] = -1
        #         b[i] = edge.weight
        # A = np.array(A)
        # b = np.array(b).reshape(-1, 1)
        # # Solve the system of equations outside the for loop
        # x , residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # x = x.flatten()   
        # x = dict(zip(self.nodesID, x))
        # return x
    
    def PolyValIteration(self, filename):
        newfile = os.path.join('Polyhedral Value Iteration', os.path.basename(filename))
        with open(newfile, 'w') as file:
            iterations = 0
            if self.nodes:
                iterations += 1
                self.initX()
                tight = self.obtainTightEdges(None)
                delta = self.calculate_Delta(tight)
                xplus = self.calculate_characterisation(delta) 
                while self.stronglyViolatingCheck(delta):
                    epsilon = self.compute_EpsilonMax(xplus)
                    xplus = self.updateShift(epsilon, xplus) ## now xplus is now epsilon * X^+
                    iterations += 1
                    self.x = self.realiseGraph(self.obtainTightEdges(xplus)) ## update our polhedron
                    tight = self.obtainTightEdges(None)
                    delta = self.calculate_Delta(tight)
                    xplus = self.calculate_characterisation(delta)
            Wmax = set()
            Wmin = set()
            all_node_ids = sorted(set(self.trivialsMin + self.trivialsMax + [node.node_id for node in self.nodes]))
            energy_value = 0
            file.write("Iteration Count: " + str(iterations) + "\n")
            for node_id in all_node_ids:
                if node_id in self.trivialsMin:
                    energy_value = 0  # Assuming trivialMin nodes have an energy value of 0
                elif node_id in self.trivialsMax:
                    energy_value = float('inf')  # Assuming trivialMax nodes have an energy value of inf
                else:
                    node = self.nodesList[node_id]
                    if xplus[node.node_id] and xplus[node.node_id] > 0:
                        Wmax.add(node)
                        energy_value = float('inf')
                        print(str(node), " Wins for Max with energy value: Infinity")
                    else:
                        Wmin.add(node)
                        energy_value = 0
                        print(str(node), " Wins for Min with energy value: ", self.x[node.node_id])
                whoWins = ''
                if energy_value != float('inf'):
                    whoWins = 'Min'
                else:
                    whoWins = 'Max'
                # Write to file (and optionally print) the node ID and its energy value
                file.write(f"{node_id} Wins for: {whoWins}\n")
                print(f"Node {node_id} Wins for: {whoWins}")
        return
    def __repr__(self):
        return f"Graph with {len(self.nodes)}, nodes and {len(self.edges)} edges"
    



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

# file_path = os.path.join('OinkBipartiteEGs', "bvb193_EnergyTest.txt")
# graph = createGraph(file_path)
# print(graph)
# for node in graph.nodes:
#     print(node)
#     node.printEdges()

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
        graph.PolyValIteration(file_path)
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
graph.PolyValIteration(file_path)