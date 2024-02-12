import os
class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type  # 'Min' or 'Max'
        self.edges = []  # List of edges connected to this node
        self.incidents = [] # List of nodes incident to this node

    def add_edge(self, edge):
        self.edges.append(edge)
    def add_incident(self, node):
        self.incidents.append(node)

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

class SCC:
    def __init__(self):
        self.nodes = []
        self.contains_max = False
        self.contains_min = False
        self.connected_sccs = []  # SCCs that this SCC has edges to

    def add_node(self, node):
        self.nodes.append(node)
        if node.node_type == 'Max':
            self.contains_max = True
        elif node.node_type == 'Min':
            self.contains_min = True

    def add_connected_scc(self, scc):
        if scc not in self.connected_sccs:
            self.connected_sccs.append(scc)

    def __repr__(self):
        node_ids = [node.node_id for node in self.nodes]
        return f"SCC(Nodes: {node_ids}, Max: {self.contains_max}, Min: {self.contains_min})"
    
    def extractTrivials(self):
        trivialnodes = []
        if (self.contains_max and self.contains_min == False):
            isTrivial = singleTypeSCCtrivials(self, self.contains_max)
            if isTrivial:
                return self.nodes
        else:
            minList, maxList = dualTypeSCCtrivials(self)
            if minList:
                trivialnodes.extend(minList)
            if maxList:
                trivialnodes.extend(maxList)
            return trivialnodes
        return trivialnodes
    


    def bellman_ford(self, start_node_id, isMaxOnly):
        V = len(self.nodes)
        dist = {node.node_id: float('inf') for node in self.nodes}
        dist[start_node_id] = 0
        predecessors = {node.node_id: None for node in self.nodes}

        # Prepare edges based on node type
        edges = []
        for node in self.nodes:
            for edge in node.edges:
                # For Min instance
                if not isMaxOnly and node.node_type == 'Min' and edge.to_node.node_type == 'Min':
                    edges.append((edge.from_node.node_id, edge.to_node.node_id, edge.weight))
                # For Max instance
                elif isMaxOnly and node.node_type == 'Max' and edge.to_node.node_type == 'Max':
                    edges.append((edge.from_node.node_id, edge.to_node.node_id, -edge.weight))  # Invert weight

        # Bellman-Ford algorithm
        for _ in range(V-1):
            for u, v, w in edges:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    predecessors[v] = u
        
        # Check for negative cycles
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return self.reconstruct_negative_cycle(predecessors, v)

        return None

    def reconstruct_negative_cycle(self, predecessors, start):
        cycle = []
        current = start
        for _ in range(len(self.nodes)):
            current = predecessors[current]
        cycle_start = current
        while True:
            cycle.append(current)
            if current == cycle_start and len(cycle) > 1:
                break
            current = predecessors[current]
        cycle.reverse()
        return cycle



    def dualTypeSCCtrivials(self):
        """ Here we want to run two instances of Bellman-ford on our SCC with a set criteria:
         the first instance of BMF will be for finding trivial cycles for min
         and analgously the second instance of bellman-ford will be for finding trivial cycles for Max
           
        NOTE this SCC is comprsied of both Min and Max nodes thus altering of the SCC is requried:
        
        For our Min instance all edges which go form Min -> Min remain the same weight. Every other edge is +inf in weight
        For our Max instance all edges which go from Max -> Max are the inverted weight. Every other edge is +inf in weight.
        IN each instance if there exists a negative cycle we wish to obtain this negative cycle.
        We return two variables, MinList and MaxList which are lists of negative cycles of their respective BMF instance if they exist.
        If only MaxList exists and MinList does not return None, MaxList. Analgously do the same in the converse situtation.   """
        start_node_id = self.nodes[0].node_id
        MinList = self.bellman_ford(start_node_id, isMaxOnly=False)
        MaxList = self.bellman_ford(start_node_id, isMaxOnly=True)

        return MinList, MaxList



    def singleTypeSCCtrivials(self, isMaxOnly):
        """ We want to do Karps Algorithm here, to find out if any negative cycles can be made
            IF so then SCC entirety is trivial and thus all nodes inside are trivial
            Else not 
            We do this dependant on ownership of the SCC, if max owns then we invert all edges before committing to Karps Algorithm
        """
        
        V = len(self.nodes)
        # Map node IDs to indices for easier access in the dp array
        node_index = {node.node_id: idx for idx, node in enumerate(self.nodes)}
        
        # Initialize the dynamic programming table
        dp = [[float('inf')] * V for _ in range(V + 1)]
        dp[0][0] = 0  # Starting from any node, 0 cost to reach itself with 0 edges
        
        # Building the DP table
        for k in range(1, V + 1):
            for i, node in enumerate(self.nodes):
                for edge in node.edges:
                    if edge.to_node.node_id in node_index:  # Ensure edge is within SCC
                        j = node_index[edge.to_node.node_id]
                        # Invert edge weight if the SCC is owned by Max nodes
                        edge_weight = -edge.weight if isMaxOnly else edge.weight
                        if dp[k-1][node_index[edge.from_node.node_id]] != float('inf'):
                            dp[k][j] = min(dp[k][j], dp[k-1][node_index[edge.from_node.node_id]] + edge_weight)
        
        # Finding the minimum mean weight cycle
        min_mean_weight = float('inf')
        for v in range(V):
            max_mean_weight = float('-inf')
            for k in range(1, V + 1):
                if dp[k][v] != float('inf'):
                    mean_weight = max(max_mean_weight, (dp[V][v] - dp[k][v]) / (V - k))
            min_mean_weight = min(min_mean_weight, mean_weight)
        
        # A negative minimum mean weight indicates a trivial SCC for both Max and Min
        isTrivial = min_mean_weight < 0
        return isTrivial


class Graph:
    def __init__(self):
        self.nodesList = {}  # Dictionary to store nodes by their ID
        self.nodes = []
        self.edges = []  # List to store all edges
        self.trivials = {}  # map to keep the trivial nodes and the winner of that respective node.
    def add_node(self, node):
        """Add a node to the graph."""
        if node.node_id not in self.nodesList:
            self.nodesList[node.node_id] = node
            self.nodes.append(node)
    def add_edge(self, from_node_id, to_node_id, weight):
        """Add an edge to the graph."""
        if from_node_id in self.nodesList and to_node_id in self.nodesList:
            from_node = self.nodesList[from_node_id]
            to_node = self.nodesList[to_node_id]
            edge = Edge(from_node, to_node, weight)
            self.edges.append(edge)
            to_node.add_incident(from_node)
            from_node.add_edge(edge)

    def remove_edge(self, from_node_id, to_node_id):
        """Remove an edge from the graph."""
        self.edges = [edge for edge in self.edges if edge.from_node.node_id != from_node_id or edge.to_node.node_id != to_node_id]
        if from_node_id in self.nodesList:
            from_node = self.nodesList[from_node_id]
            from_node.edges = [edge for edge in from_node.edges if edge.to_node.node_id != to_node_id]
        if to_node_id in self.nodesList:
            to_node = self.nodesList[to_node_id]
            to_node.incidents = [node for node in to_node.incidents if node.node_id != from_node_id]

    def remove_node(self, node_id):
        """Remove a node and its associated edges from the graph."""
        if node_id in self.nodesList:
            node_to_remove = self.nodesList[node_id]
            
            # Remove outgoing edges
            for edge in node_to_remove.edges[:]:  # Iterate over a copy of the list
                self.remove_edge(node_id, edge.to_node.node_id)
            
            # Remove incoming edges
            for incident_node in node_to_remove.incidents[:]:  # Iterate over a copy of the list
                self.remove_edge(incident_node.node_id, node_id)

            # Finally, remove the node from the graph's node list and dictionary
            del self.nodesList[node_id]
            self.nodes = [node for node in self.nodes if node.node_id != node_id]
            # Also, update the edges list to ensure all references to the removed node are cleared
            self.edges = [edge for edge in self.edges if edge.from_node.node_id != node_id and edge.to_node.node_id != node_id]

    def __repr__(self):
        return f"Graph with {len(self.nodes)}, nodes and {len(self.edges)} edges"
    
    def tarjan_scc(self):
        """Perform Tarjan's SCC decomposition and return a list of SCC objects."""
        self.index = 0
        self.stack = []
        self.indices = {}  # Dictionary to store the index of each node
        self.low_links = {}  # Dictionary to store the lowest index reachable from each node
        self.on_stack = {}  # Dictionary to track if a node is in the stack
        self.sccs = []  # List to store the resulting SCCs

        def strongconnect(node):
            """Helper function for the strongly connected components algorithm."""
            self.indices[node.node_id] = self.index
            self.low_links[node.node_id] = self.index
            self.index += 1
            self.stack.append(node)
            self.on_stack[node.node_id] = True

            # Explore successors of the node
            for edge in node.edges:
                successor = edge.to_node
                if successor.node_id not in self.indices:
                    strongconnect(successor)
                    self.low_links[node.node_id] = min(self.low_links[node.node_id], self.low_links[successor.node_id])
                elif self.on_stack[successor.node_id]:
                    self.low_links[node.node_id] = min(self.low_links[node.node_id], self.indices[successor.node_id])

            # If node is a root node, pop the stack and generate an SCC
            if self.low_links[node.node_id] == self.indices[node.node_id]:
                new_scc = SCC()
                while True:
                    successor = self.stack.pop()
                    self.on_stack[successor.node_id] = False
                    new_scc.add_node(successor)
                    if successor == node:
                        break
                self.sccs.append(new_scc)

        # Initialize the algorithm
        for node in self.nodes:
            if node.node_id not in self.indices:
                strongconnect(node)

        # At this point, self.sccs contains all the identified SCCs
        # Now, determine connections between SCCs
        self._determine_scc_connections()

        return self.sccs

    def _determine_scc_connections(self):
        """Determine connections between identified SCCs."""
        for scc in self.sccs:
            for node in scc.nodes:
                for edge in node.edges:
                    to_node_scc = self._find_scc_of_node(edge.to_node)
                    if to_node_scc and to_node_scc != scc:
                        scc.add_connected_scc(to_node_scc)

    def _find_scc_of_node(self, node):
        """Find which SCC a node belongs to."""
        for scc in self.sccs:
            if node in scc.nodes:
                return scc
        return None

    def BFSattractors(self,alltrivs):
        """ This function takes a list of tuples and outputs another list of tuples
         The procedure:
        This function will act as an inverted BFS, so we travel to nodes if they have an incident edge to the current node we are on. 
        This is acting as our attractors. 
        we initialise two lists, trivials and alltrivs, where trivials is our result array and alltrivs is our queue.
        for all nodes in our queue we do the following:
        pop from queue.
        add to trivials
        obtain all incident nodes. 
        If the tuples second element is min:
        if any incidents are min, add them to alltrivs with the tuple (node, min)
        if any are max, check if all edges outgoing go to a node in self.trivials with value min or to a dictionary which stores all of the seen trivials so far in our BFS with value min
            if all edges go to a node with min trivial value then make the tuple node, min and add to our queue.

        This is analagous for if the second element in our tuple is max.
        """
        trivials = []  # Result array, will contain tuples of (Node object, 'min'/'max')
        queue = alltrivs[:]  # Initialize queue with alltrivs
        seen_trivials = {node.node_id: node_type for node, node_type in alltrivs}  # Track seen trivial nodes by their ID

        while queue:
            current_node, node_type = queue.pop(0)  # Dequeue the next node
            if current_node.node_id not in seen_trivials:  # Process if not already seen
                trivials.append((current_node, node_type))  # Mark as trivial
                seen_trivials[current_node.node_id] = node_type  # Track seen trivial nodes

                # Process incident nodes
                for incident_node in current_node.incidents:
                    incident_node_type = incident_node.node_type

                    # Process based on current trivial node type
                    if node_type == 'Min':
                        if incident_node_type == 'Min':
                            if incident_node.node_id not in seen_trivials:
                                queue.append((incident_node, 'Min'))
                                seen_trivials[incident_node.node_id] = 'Min'
                        elif incident_node_type == 'Max':
                            if all((edge.to_node.node_id in seen_trivials and seen_trivials[edge.to_node.node_id] == 'Min') for edge in incident_node.edges):
                                if incident_node.node_id not in seen_trivials:
                                    queue.append((incident_node, 'Min'))
                                    seen_trivials[incident_node.node_id] = 'Min'
                    
                    elif node_type == 'Max':
                        if incident_node_type == 'Max':
                            if incident_node.node_id not in seen_trivials:
                                queue.append((incident_node, 'Max'))
                                seen_trivials[incident_node.node_id] = 'Max'
                        elif incident_node_type == 'Min':
                            if all((edge.to_node.node_id in seen_trivials and seen_trivials[edge.to_node.node_id] == 'Max') for edge in incident_node.edges):
                                if incident_node.node_id not in seen_trivials:
                                    queue.append((incident_node, 'Max'))
                                    seen_trivials[incident_node.node_id] = 'Max'
        return trivials

    def removeAllTrivials(self):
        """ Aim of this function is to remove all Trivial Nodes and cycles found in our graph.
            This is done by calling .extractTrivials() on all our SCCs and merging all the resultant lists, k .
            If this list is non-empty then we can make use of attracters to conclusively trivially find other nodes we know win for Max or Min
            This is done by conducting a BFS: 
            If the node in K is owned by min, Then if any of its incident min-nodes are also trivial and should be added to our list.
            If any of its incident Max-nodes can only go to nodes in our trivials dictionary winning for min or trivial min nodes in K, then that node wins for min and should be added to K
            this is analgous for max nodes in K. 
            Otherwise if our initialisation of K is empty we are done and all trivial nodes/cycles have been removed. 
            If k was not empty, after conducting our reverse BFS like traversal.
            We remove all nodes that were in K, add them to our trivials dictionary and recompute the SCCs. Recursively calling this algorithm until there are no more trivial nodes and cycles"""
        alltrivs = []
        for scc in self.sccs:
            trivial_nodes = scc.extractTrivials()  # Returns a list of Node objects
            # Transform trivial_nodes into a list of tuples (node_id, node_type)
            for node in trivial_nodes:
                alltrivs.append((node, node.node_type))
        while alltrivs:
            alltrivs = self.BFSattractors(alltrivs)
            for node, node_val in alltrivs:
                self.remove_node(node)
                self.trivials[node] = node_val

            # Recompute SCCs to reflect changes in the graph
            self.tarjan_scc()
            alltrivs = []
            for scc in self.sccs:
                trivial_nodes = scc.extractTrivials()  # Returns a list of Node objects
                # Transform trivial_nodes into a list of tuples (node_id, node_type)
                for node in trivial_nodes:
                    alltrivs.append((node, node.node_type))
        return

def createGraph(filename):
    graph = Graph()
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Check if line is not empty
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
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Check if line is not empty
                parts = line.strip().split(' ')
                identifier = int(parts[0])
                type = int(parts[1])
                successors = parts[2].split(',')
                successors = list(map(int, successors))
                weights = parts[3].split(',')
                weights = list(map(int, weights))
                for successor , weight in zip(successors,weights):
                    graph.add_edge(identifier, successor, weight)

    return graph

graph = createGraph(os.path.join('OinkEGtests', "vb054_EnergyTest.txt"))
print(graph)
for node in graph.nodes:
    print(node)
    node.printEdges()

# # Initialize the Graph
# graph = Graph()

# # Add nodes with their IDs and types ('Min' or 'Max')
# graph.add_node(Node('A', 'Min'))
# graph.add_node(Node('B', 'Min'))
# graph.add_node(Node('C', 'Max'))
# graph.add_node(Node('D', 'Max'))

# # Add edges between nodes
# graph.add_edge('A', 'B', 1)  # Edge from A to B with weight 1
# graph.add_edge('B', 'C', 2)  # Edge from B to C with weight 2
# graph.add_edge('C', 'A', 3)  # Edge from C to A with weight 3, forming a cycle A -> B -> C -> A
# graph.add_edge('C', 'D', 4)  # Edge from C to D with weight 4
# # Assuming D does not link back to A, B, or C, it forms its own SCC

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

sccs = graph.tarjan_scc()

print("Identified SCCs:")
for scc in sccs:
    print(scc)