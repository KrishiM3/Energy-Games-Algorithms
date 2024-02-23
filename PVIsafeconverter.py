import os
import random

def process_parity_game(input_filepath, output_filepath):
    # Read the input file and process each line
    with open(input_filepath, 'r') as file_in, open(output_filepath, 'w') as file_out:
        for line in file_in:
            if line.strip() and not line.startswith('parity'):
                parts = line.split(' ')
                # Check if the priority exceeds 6 and replace it if needed
                priority = int(parts[1])
                if priority > 6:
                    parts[1] = str(random.randint(0, 6))
                # Write the updated line to the output file
                file_out.write(' '.join(parts))
            else:
                # Write the line as is if it's a comment or empty
                file_out.write(line)

def process_all_files(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Construct full paths for input and output files
        input_filepath = os.path.join(input_dir, filename)
        output_filename = 'p' + filename  # Prefix the filename with 'p'
        output_filepath = os.path.join(output_dir, output_filename)

        # Process each file
        process_parity_game(input_filepath, output_filepath)
        print(f"Updated file has been written to: {output_filepath}")

# Define the base directory for input and output
input_subdir = 'OinkPGtests'
output_subdir = 'PVIsafeOinkPGs'

# Process all files in the input directory
process_all_files(input_subdir, output_subdir)