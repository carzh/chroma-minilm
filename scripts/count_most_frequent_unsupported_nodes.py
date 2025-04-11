import os 
import argparse

def process_file(file_path):
    # Read file and handle errors
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None

    # Processing Step 1: Remove lines ending with [1]
    step1_lines = [line for line in lines if not line.rstrip().endswith('[1]')]
    print("\nAfter removing lines ending with [1]:")
    print(''.join(step1_lines))

    # Processing Step 2: Remove lines without "IsOpSupported"
    step2_lines = [line for line in step1_lines if "IsOpSupported" in line]
    print("\nAfter keeping only lines with 'IsOpSupported':")
    print(''.join(step2_lines))

    # Define target operations
    target_ops = ["Concat", "Reshape", "Transpose"]
    
    # Count occurrences
    op_counts = {op: 0 for op in target_ops}
    for line in step2_lines:
        for op in target_ops:
            op_counts[op] += line.count(op)
    
    # Sort by frequency
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nOperation counts:")
    for op, count in sorted_ops:
        print(f"{op}: {count}")
    
    return sorted_ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count most frequent unsupported nodes in CoreML model from segment of verbose output')
    parser.add_argument('file_path', type=str, help='Path to verbose output file')
    args = parser.parse_args()
    
    process_file(args.file_path)