import os

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        f1_lines = f1.readlines()[1:]  # Skip the first line
        f2_lines = f2.readlines()[1:]  # Skip the first line

        for line1, line2 in zip(f1_lines, f2_lines):
            # Replace 'Infinity' with 'inf' for comparison
            line1 = line1.replace('Infinity', 'inf')
            line2 = line2.replace('Infinity', 'inf')

            # print(line1)
            # print(line2)

            if line1 != line2:
                # print(line1, line2)
                return False
    return True

def main():
    gkk_dir = 'GKK'
    fvi_dir = 'Polyhedral Value Iteration'


    for filename in os.listdir(gkk_dir):
        gkk_file = os.path.join(gkk_dir, os.path.basename(filename))
        fvi_file = os.path.join(fvi_dir, os.path.basename(filename))

        if os.path.isfile(gkk_file) and os.path.isfile(fvi_file):
            if not compare_files(gkk_file, fvi_file):
                print(f"{filename} is different")
        else:
            print(f"File missing: {filename}")

if __name__ == "__main__":
    main()
