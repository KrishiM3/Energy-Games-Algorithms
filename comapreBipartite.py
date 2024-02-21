import os
from difflib import unified_diff

# Function to compare two files, ignoring the first line
def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Read lines and ignore the first line
        lines1 = f1.readlines()[1:]
        lines2 = f2.readlines()[1:]
    
    # Use unified_diff to compare the files line by line
    diff = list(unified_diff(lines1, lines2, fromfile=file1, tofile=file2))
    if diff:
        print(f"Differences found between {file1} and {file2}:")
        for line in diff:
            print(line, end='')
    else:
        print(f"No differences found between {file1} and {file2}.")

# Path to the directory containing the files
directory_path = "Fast Value Iteration"

# List all files in the directory
files = os.listdir(directory_path)

# Identify pairs of files to compare
pairs_to_compare = []
for file in files:
    if file.startswith("b") and file.endswith("_EnergyTest.txt"):
        # Construct the corresponding Y_EnergyTest.txt filename
        corresponding_file = file[1:]
        if corresponding_file in files:
            pairs_to_compare.append((os.path.join(directory_path, file), 
                                     os.path.join(directory_path, corresponding_file)))

# Compare each pair of files
for file1, file2 in pairs_to_compare:
    compare_files(file1, file2)