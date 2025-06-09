#!/usr/bin/env python3
# filepath: analyze_unsupported_operators.py

import sys
import re
from collections import Counter, defaultdict
import argparse
from typing import Dict, List, Tuple, DefaultDict
import os


def parse_unsupported_operators(file_path: str) -> Tuple[Counter, DefaultDict[str, Counter]]:
    """
    Parse the ONNX Runtime log file and extract information about unsupported operators and their reasons.
    Uses a more general approach to detect any reason patterns.
    
    Args:
        file_path: Path to the ONNX Runtime log file
        
    Returns:
        Tuple containing:
        - Counter of unsupported operator types
        - Dictionary mapping operator types to Counter of reasons for unsupported status
    """
    unsupported_ops = Counter()
    unsupported_reasons = defaultdict(Counter)
    
    current_op = None
    potential_reason = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect when an operator is found to be unsupported
            op_not_supported = re.search(r'Operator type: \[(\w+)\].+supported: \[0\]', line)
            if op_not_supported:
                op_type = op_not_supported.group(1)
                unsupported_ops[op_type] += 1
                
                # If we found a reason in a previous line, associate it with this operator
                if potential_reason:
                    unsupported_reasons[op_type][potential_reason] += 1
                    potential_reason = None
                else:
                    unsupported_reasons[op_type]["No explicit reason given"] += 1
                
                continue
            
            # Look for lines that indicate an operation is not supported with a specific reason
            reason_indicator = re.search(r'\[\w+\] (has|is not|does not|failing)', line)
            if reason_indicator:
                # Extract the operator name if present
                op_match = re.search(r'\[(\w+)\]', line)
                if op_match:
                    current_op = op_match.group(1)
                
                # Extract the reason from the line
                # First try to get the full message after the operator
                if current_op:
                    pattern = rf'\[{current_op}\] (.+)'
                    reason_match = re.search(pattern, line)
                    if reason_match:
                        potential_reason = reason_match.group(1).strip()
                        continue
            
            # Match other common reason patterns
            reason_patterns = [
                # Check for 'not supported' patterns
                r'does not support (.+)',
                r'is not (a |currently )?supported',
                r'must be (.+)',
                r'failing (.+)',
                # Check for dynamic dimension issues
                r'has shape with dynamic dimension',
                # Check for initializer issues
                r'\'([^\']+)\'.+is not a constant initializer',
                # Check for CoreML specific limitations
                r'CoreML does not support (.+)'
            ]
            
            for pattern in reason_patterns:
                match = re.search(pattern, line)
                if match:
                    if match.groups():
                        potential_reason = match.group(0).strip()
                    else:
                        potential_reason = pattern.strip()
                    break
    
    return unsupported_ops, unsupported_reasons


def print_results(unsupported_ops: Counter, unsupported_reasons: DefaultDict[str, Counter]) -> None:
    """Print formatted results of the analysis."""
    print(f"{'=' * 80}")
    print(f"UNSUPPORTED OPERATOR ANALYSIS")
    print(f"{'=' * 80}\n")
    
    total_unsupported = sum(unsupported_ops.values())
    print(f"Total unsupported operator instances: {total_unsupported}")
    print(f"Unique unsupported operator types: {len(unsupported_ops)}\n")
    
    print(f"Unsupported operator types by frequency:")
    print(f"{'-' * 40}")
    
    for op, count in sorted(unsupported_ops.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_unsupported) * 100
        print(f"{op}: {count} ({percentage:.1f}%)")
    
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
    
    print(f"{'=' * 80}")
    print(f"AGGREGATED REASONS ACROSS ALL OPERATORS")
    print(f"{'=' * 80}\n")
    
    # Aggregate all reasons across operators
    all_reasons = Counter()
    for op, reasons in unsupported_reasons.items():
        all_reasons.update(reasons)
    
    for reason, count in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_unsupported) * 100
        print(f"{reason}: {count} ({percentage:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ONNX Runtime unsupported operators and reasons")
    parser.add_argument("log_file", help="Path to the ONNX Runtime log file")
    parser.add_argument("--output", "-o", help="Output file to save results (optional)")
    args = parser.parse_args()
    
    try:
        print(f"Analyzing file: {args.log_file}")
        print(f"File size: {os.path.getsize(args.log_file) / 1024:.2f} KB")
        
        unsupported_ops, unsupported_reasons = parse_unsupported_operators(args.log_file)
        
        # Redirect output to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                original_stdout = sys.stdout
                sys.stdout = f
                print_results(unsupported_ops, unsupported_reasons)
                sys.stdout = original_stdout
                print(f"Results saved to {args.output}")
        else:
            print_results(unsupported_ops, unsupported_reasons)
            
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()