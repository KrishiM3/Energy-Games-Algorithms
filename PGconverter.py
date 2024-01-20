import os

class Node:
    def __init__(self,id, priority, type, successors):
        self.id = id
        self.priority = priority
        self.type = type
        self.successors = successors

def parse_line(line):
    parts = line.strip().split(' ')
    identifier = int(parts[0])
    priority = int(parts[1])
    owner = int(parts[2])
    successors = list(map(int, parts[3].split(',')))
    
    return Node(identifier,priority,owner, successors)

def convertPGfile(filename):
    nodeDict = dict()
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            if line.strip():  # Check if line is not empty
                newnode = parse_line(line)
                nodeDict[newnode.id] = newnode
    newfile = '/OinkEGtests/' + os.path.splitext(filename)[0] + '_EnergyTest.txt'
    m = len(nodeDict)
    with open(newfile, 'w') as file:
        for key in nodeDict:
            node = nodeDict[key]
            edges = []
            weights = []
            for successor in node.successors:
                edges.append(str(successor))
                weights.append(str((-m) ** nodeDict[successor].priority))
            # Convert lists to comma-separated strings
            edges_str = ','.join(edges)
            weights_str = ','.join(weights)

            # Write formatted line to file
            file.write(f"{node.id} {node.type} {edges_str} {weights_str}\n")
    return 

# Parse the content of the simulated 'test.txt' file
parsed_data = convertPGfile('/OinkPGtests/vb001')
