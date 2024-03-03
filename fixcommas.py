import os

def clean_line(line):
    # Function to clean a single line except the first line of the file
    parts = line.strip().split()
    # Ensure that commas are properly placed without duplication
    if len(parts) > 3:  # If there are elements to join with commas
        cleaned = ' '.join(parts[:3]) + ' ' + ','.join(filter(None, parts[3:])).strip(';') + ';'
    else:  # If there are not enough elements to require commas
        cleaned = ' '.join(parts).strip(';') + ';'
    return cleaned + '\n'

def clean_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Preserve the first line and clean the rest
    first_line = lines[0]
    cleaned_lines = [first_line] + [clean_line(line) for line in lines[1:] if line.strip()]
    
    # Overwrite the file with cleaned content
    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)

def clean_all_files_in_directory(directory):
    iter = 0
    for filename in os.listdir(directory):
        if filename.endswith(".gm"):  # Ensure you're processing .gm files
            file_path = os.path.join(directory, filename)
            clean_file(file_path)
            iter += 1
            print(f"Cleaned {filename}")

# Replace 'your_directory_path' with the actual path to your directory
directory = 'equivchecking'  # Update this to your directory
clean_all_files_in_directory(directory)