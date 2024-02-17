import heapq
import os
class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type  # 'Min' or 'Max'
        self.edges = []  # List of edges connected to this node
        self.incidents = [] # List of nodes incident to this node
        self.totalPotential = 0
        self.prevPotential = 0
        self.heapValue = float("inf") ## init to infinity for calculating Enplus values.

    def add_edge(self, edge):
        self.edges.append(edge)
    def add_incident(self, node):
        self.incidents.append(node)
    def __lt__(self, other):
        return self.heapValue < other.heapValue

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
            to_node.add_incident(from_node)
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
            # self.set_potential(node, 0)
            node.heapValue = float("inf") ## resets heapvalues also. 
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
            adjusted_weight = 0
            if self.potentials[edge.to_node.node_id] == float("inf") and self.potentials[edge.from_node.node_id] == float("inf"):
                adjusted_weight = edge.weight
            else:
                adjusted_weight = edge.weight + self.potentials[edge.to_node.node_id] - self.potentials[edge.from_node.node_id]
            # Store the modified weight with a tuple of the edge's start and end node IDs
            modified_weights[edge] = adjusted_weight
        return modified_weights

    def calculate_EnPlus(self, modified_weights):
        F = set()
        maxSet = set() ## set of max nodes not in F
        minSet = set() ## set of min nodes not in F
        EnPlus = {} ## dictionary for all the ENplus vals 
        # for node in self.nodes:
        #     EnPlus[node] = -float("inf")
        heap = [] ## pq for iter 2
        # heapq.heappop(heap)[1]
        counters = {} ## counter of max for iter 1 
        zeros = set() ## set denoting 0 counts for iter 1
        translatedWeights = {(edge.from_node, edge.to_node): value for edge, value in modified_weights.items()}
        # This is broken into two steps, first initialise our set F, then apply our iteration of the variation of dijkstras.
        # to init F, if node is min and has a single negative edge we add it
        #  if node is max and all edges are negative then add to F.
        
        for node in self.nodes:
            isF = False
            if node.node_type == 'Min':
                for edge in node.edges:
                    if (modified_weights[edge] < 0):
                        isF = True
                        break
            else:
                isF = True
                for edge in node.edges:
                    if (modified_weights[edge] >= 0):
                        isF = False
                        break
            
            if isF is True:
                F.add(node)
                EnPlus[node] = 0
        
        ## create our max and min sets 
        for node in self.nodes:
            if node in F:
                continue
            if node.node_type == 'Min': ## init our heap value with inf
                minSet.add(node)
                heapq.heappush(heap, (node.heapValue, node))

            else: ## calculate the count of edges of our node and populate zeros and counters correctly
                count = 0
                for edge in node.edges:
                    if edge.to_node in F:
                        continue
                    elif modified_weights[edge] >= 0:
                        count = count + 1
                    else:
                        pass
                maxSet.add(node)
                if count == 0:
                    zeros.add(node)
                else:
                    counters[node] = count

        ## now we need to initialise our heap with the correct minimisations, only using edges from Vmin/ F to F 
        ## we also know at the beginning all nodes in F will have En+ as 0. so this is simply the smallest weights. 
                    
        for node in minSet:
            currMin = float("inf")
            for edge in node.edges:
                if edge.to_node in F:
                    currMin = min(currMin, modified_weights[edge])
            node.heapValue = currMin
            heapq.heappush(heap, (node.heapValue, node))
        ## step one complete, now for step two:
        ''' Two iterations:
            1 - go through Vmax not in F and if there exists a vertex which 
            all non-negative edges go to F then set En^+(v) = max (w(vv') + En^+(v))
            
            2 - Otherwise, go through Vmin not in F which has edges to F and find the edge with minimises the above equation,
            min (w(vv') + En^+(v)), then set the vertex with that Enplus value.

        Problem lies with doing this efficiently, it is stated we can use dijkstras and thus obtain a mlogn solution. 


        Initial thoughts, keep a count of the non-negative edges of each Vmax which do not go F, 
        if this is 0 we can perform our brute force O(M) check on it to get max (w(vv') + En^+(v)) 

        When we add a vertex to F we can check if each Vmax not in F has an edge to it, if so subtract the count by 1. 
        can be done in O(M) if we use dictionaries for each node holding their incident nodes. 

        That settles iteration 1. 

        For iteration 2, clearly dijkstra's should be used as we have not used it for iter 1. 

        Before we start doing iteration 1 it makes sense to initalise a min heap, of all Vmin vertices not in F with the nodes being 
        only populated by the weights of edges from those to a node in F. This can be done in O(M). 
        if we are in iter 2, simply add the min node from the heap to F, (check if the weight is non-negative if so then terminate iter 2). 
        Then when we add a vertex to F from iter 1 or 2:

        if it is Max:
            then with all incident nodes to it, if one of them is Vmin not in F, then update the heap accordingly 
        if it is Min:
            remove the node from the heap (this will just be the top of the heap by iter 2 then...),
            then with all incident nodes to it, if one of them is Vmin not in F, then update the heap accordingly

        go back to iter 1. 

        '''
        while True:
            # iter 1. 
            if zeros:
                elem = next(iter(zeros)) # obtain any node with count of 0
                zeros.remove(elem) # remove from zero set
                F.add(elem) # add to F now
                maxSet.remove(elem) # remove from set of max not in F
                for incident in elem.incidents: 
                    if translatedWeights[(incident,elem)] >= 0:
                        currCount = counters.pop(incident, 0) # get the count of the nodes incident to it and reduce by 1
                        currCount = currCount - 1
                        if currCount == 0:
                            zeros.add(incident) 
                        else:
                            if currCount > 0:
                                counters[incident] = currCount

                ## now set the value of En^+ of the vertex with 0 count
                nodemax = -float("inf")
                for edge in elem.edges:
                    if  edge.to_node in EnPlus:
                        nodemax = max(modified_weights[edge] + EnPlus[edge.to_node], nodemax)
                EnPlus[elem] = nodemax
                for incident in elem.incidents:
                    if incident in minSet:
                        before = incident.heapValue
                        incident.heapValue = min(incident.heapValue, translatedWeights[(incident,elem)] + EnPlus[elem] )
                        if before != incident.heapValue:
                            heapq.heappush(heap, (incident.heapValue, incident))
            else:
                elem = None
                while (elem in F or elem is None) and heap:
                    elem = heapq.heappop(heap)[1]
                if not heap:
                    break
                if elem is None:
                    break
                if elem.heapValue == float("inf"):
                    minSet.add(elem)
                    break
                else:
                    F.add(elem)
                    minSet.remove(elem)
                    EnPlus[elem] = elem.heapValue
                    for incident in elem.incidents:
                        if incident.node_type == "Max" and incident in maxSet and translatedWeights[(incident,elem)] >= 0:
                            currCount = counters.pop(incident, 0) # get the count of the nodes incident to it and reduce by 1
                            currCount = currCount - 1
                            if currCount == 0:
                                zeros.add(incident) 
                            else:
                                if currCount > 0:
                                    counters[incident] = currCount
                        else:
                            if incident.node_type == "Min" and incident in minSet:
                                before = incident.heapValue
                                incident.heapValue = min(incident.heapValue, translatedWeights[(incident,elem)] + elem.heapValue )
                                if before != incident.heapValue:
                                    heapq.heappush(heap, (incident.heapValue, incident))
                    
            
        ## now broken out, all remaining vertices have EN+ = inf
        for node in maxSet:
            EnPlus[node] = float("inf")
        for node in minSet:
            EnPlus[node] = float("inf")
        return EnPlus
        
    def existsChange(self,EnPlus):
        change = False
        for node in self.nodes:
            if node.totalPotential != node.totalPotential + EnPlus[node]:
                change = True
        return change
    
    def FVI(self,filename):
        iteration = 0
        while True:
            modified_weights = self.calculate_modified_weights()
            EnPlus = self.calculate_EnPlus(modified_weights)
            if not self.existsChange(EnPlus):
                break
            self.apply_modified(modified_weights)
            for node in self.nodes:
                self.set_potential(node, EnPlus[node])
            self.reset_potentials()
            print(iteration)
            iteration = iteration + 1
        newfile = os.path.join('Fast Value Iteration', os.path.basename(filename))
        with open(newfile, 'w') as file:
            file.write("Iteration Count: " + str(iteration) + "\n")
            all_node_ids = sorted(set(self.trivialsMin + self.trivialsMax + [node.node_id for node in self.nodes]))
            for node_id in all_node_ids:
                if node_id in self.trivialsMin:
                    energy_value = 0  # Assuming trivialMin nodes have an energy value of 0
                elif node_id in self.trivialsMax:
                    energy_value = float('inf')  # Assuming trivialMax nodes have an energy value of inf
                else:
                    node = self.nodesList[node_id]
                    energy_value = node.totalPotential  # Fetch the energy value from the node object
                
                # Write to file (and optionally print) the node ID and its energy value
                file.write(f"{node_id} has energy value of: {energy_value}\n")
                print(f"Node {node_id} has energy value of: {energy_value}")
        return
        # EnPlus = {}
        # start = False
        # for node in self.nodes:
        #     EnPlus[node] = 0
        # while (not start) or (self.existsChange(EnPlus)):
        #     for node in self.nodes:
        #         self.set_potential(node, EnPlus[node])
        #     self.reset_potentials()
        #     start = True
        #     modified_weights = self.calculate_modified_weights()
        #     EnPlus = self.calculate_EnPlus(modified_weights)
        #     self.apply_modified(modified_weights)
        #     for node in self.nodes:
        #         print("Node, ", node.node_id, "has energy value, ", node.totalPotential)
        # for node in self.nodes:
        #     print("Node, ", node.node_id, "has energy value, ", node.totalPotential)
        # return

def createGraph(filename):
    print(filename)
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
                if type == 0:
                    graph.add_node(Node(identifier, 'Min'))
                else:
                    graph.add_node(Node(identifier, 'Max'))
            linenumber += 1
    with open(filename, 'r') as file:
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
                    graph.add_edge(identifier, successor, weight)
            linenumber += 1
    return graph

# graph = createGraph(os.path.join('OinkEGtests', "vb192_EnergyTest.txt"))
# print(graph)
# for node in graph.nodes:
#     print(node)
#     node.printEdges()

# """
directory = 'OinkBipartiteEGs'
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        graph = createGraph(file_path)
        print(graph)
        for node in graph.nodes:
            print(node)
            node.printEdges()
        graph.FVI(file_path)
# """
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
# graph.FVI()
