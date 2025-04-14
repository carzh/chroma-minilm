#!/usr/bin/env python3
# filepath: process_unsupported_operators.py

import sys
import re
from collections import Counter
import argparse
from typing import Dict, List, Tuple


def parse_unsupported_operators(file_path: str) -> Tuple[Counter, Dict[str, Counter]]:
    """
    Parse the ONNX Runtime log file and extract information about unsupported operators and reasons.
    
    Args:
        file_path: Path to the ONNX Runtime log file
        
    Returns:
        Tuple containing:
        - Counter of unsupported operator types
        - Dictionary mapping operator types to Counter of reasons for unsupported status
    """
    unsupported_ops = Counter()
    unsupported_reasons = {}
    current_op = None
    current_reason = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Extract operator name being checked
            op_check = re.search(r'Operator \[(\w+)\] (has|is)', line)
            if op_check:
                current_op = op_check.group(1)
                
            # Check if the operator is not supported
            not_supported = re.search(r'Operator type: \[(\w+)\] .+supported: \[0\]', line)
            if not_supported:
                op_type = not_supported.group(1)
                unsupported_ops[op_type] += 1
                
                # Ensure this operator type exists in the reasons dictionary
                if op_type not in unsupported_reasons:
                    unsupported_reasons[op_type] = Counter()
                    
                # If we have identified a reason in a previous line, associate it with this operator
                if current_reason:
                    unsupported_reasons[op_type][current_reason] += 1
                    current_reason = None
                else:
                    unsupported_reasons[op_type]["No explicit reason given"] += 1
            
            # Look for specific reasons
            # Input type not supported
            input_type_match = re.search(r'\[(\w+)\] Input type: \[(\d+)\] is not( currently)? supported', line)
            if input_type_match:
                current_op = input_type_match.group(1)
                current_reason = f"Input type {input_type_match.group(2)} is not supported"
            
            # Shape with dynamic dimension
            shape_match = re.search(r'NodeArg \[([^\]]+)\] has shape with dynamic dimension', line)
            if shape_match:
                current_reason = "Has shape with dynamic dimension"
            
            # Input dimension too large
            dim_match = re.search(r'CoreML does not support input dim > (\d+)', line)
            if dim_match:
                current_reason = f"Input dimension exceeds {dim_match.group(1)}"
            
            # Scalar indices not supported
            scalar_match = re.search(r'(\w+) does not support scalar \'indices\'', line)
            if scalar_match:
                current_reason = "Does not support scalar indices"
            
            # Constant initializer required
            const_match = re.search(r'New shape of reshape must be a constant initializer', line)
            if const_match:
                current_reason = "New shape must be a constant initializer"
                
            # Not constant initializer tensor
            not_const_match = re.search(r'\'([^\']+)\'.+is not a constant initializer tensor', line)
            if not_const_match:
                current_reason = f"'{not_const_match.group(1)}' is not a constant initializer"
                
            # Static shape check failure
            static_shape_match = re.search(r'RESHAPE failing static shape check for input', line)
            if static_shape_match:
                current_reason = "Failed static shape check"
                
    return unsupported_ops, unsupported_reasons


def print_results(unsupported_ops: Counter, unsupported_reasons: Dict[str, Counter]) -> None:
    """Print formatted results of the analysis."""
    print(f"{'=' * 80}")
    print(f"UNSUPPORTED OPERATOR ANALYSIS")
    print(f"{'=' * 80}\n")
    
    print(f"Total unsupported operators: {sum(unsupported_ops.values())}")
    print(f"\nUnsupported operator types (count):")
    print(f"{'-' * 40}")
    
    for op, count in sorted(unsupported_ops.items(), key=lambda x: x[1], reverse=True):
        print(f"{op}: {count}")
    
    print(f"\n{'=' * 80}")
    print(f"FAILURE REASONS BY OPERATOR TYPE")
    print(f"{'=' * 80}\n")
    
    for op in sorted(unsupported_reasons.keys()):
        total_count = sum(unsupported_reasons[op].values())
        print(f"{op} ({total_count} instances):")
        print(f"{'-' * 40}")
        
        for reason, count in sorted(unsupported_reasons[op].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
        
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ONNX Runtime unsupported operators and reasons")
    parser.add_argument("log_file", help="Path to the ONNX Runtime log file")
    args = parser.parse_args()
    
    try:
        unsupported_ops, unsupported_reasons = parse_unsupported_operators(args.log_file)
        print_results(unsupported_ops, unsupported_reasons)
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()