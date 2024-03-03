import os

def clean_line(line):
    # Function to clean a single line except the first line of the file
    # Replace double commas with a single comma and ensure proper termination with a semicolon
    cleaned = line.replace(',,', ',').strip()
    if not cleaned.endswith(';'):
        cleaned += ';'
    return cleaned + '\n'

def clean_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # No change needed for the first line
    # Apply clean_line function to each of the subsequent lines if they are not empty
    cleaned_lines = [lines[0]] + [clean_line(line) for line in lines[1:] if line.strip()]
    
    # Overwrite the file with cleaned content
    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)

def clean_all_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".gm"):  # Ensure you're processing .gm files
            file_path = os.path.join(directory, filename)
            clean_file(file_path)
            print(f"Cleaned {filename}")

# Replace 'your_directory_path' with the actual path to your directory
directory = 'equivchecking'  # Update this to your directory
clean_all_files_in_directory(directory)