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
    # Check if the last character of the last successor is a semi-colon and remove it
    successors = parts[3].split(',')
    if successors[-1].endswith(';'):
        successors[-1] = successors[-1][:-1]  # Remove the semi-colon from the last successor
    successors = list(map(int, successors))
    
    return Node(identifier,priority,owner, successors)

def convertPGfile(filename):
    nodeDict = dict()
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            if line.strip():  # Check if line is not empty
                newnode = parse_line(line)
                nodeDict[newnode.id] = newnode
    # Extract the base name without the path and extension
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Construct the new file path
    newfile = os.path.join('EGtests', f"{base_name}_EnergyTest.txt")
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

# Define the directory
directory = 'PGtests'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    
    # Check if it's a file and not a directory
    if os.path.isfile(file_path):
        # Call the convertPGfile function on each file
        parsed_data = convertPGfile(file_path)
        # Do something with parsed_data if needed, or continue