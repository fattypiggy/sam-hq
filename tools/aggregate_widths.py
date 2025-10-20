#!/usr/bin/env python3
"""
Aggregate width measurements from multiple subdirectories into summary CSV files.

This script reads widths.csv from all subdirectories under a root directory
and generates two aggregate CSV files:
1. max_width_px_summary.csv - Contains max_width_px from all subdirectories
2. max_width_scaled_summary.csv - Contains max_width_scaled from all subdirectories

Each column represents a subdirectory, and rows contain the measurements.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def find_width_csv_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    Find all widths.csv files in subdirectories.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of tuples (subdirectory_name, csv_file_path)
    """
    root_path = Path(root_dir)
    csv_files = []
    
    for subdir in sorted(root_path.iterdir()):
        if subdir.is_dir():
            csv_path = subdir / "widths.csv"
            if csv_path.exists():
                csv_files.append((subdir.name, str(csv_path)))
            else:
                print(f"⚠️  Warning: No widths.csv found in {subdir.name}")
    
    return csv_files


def read_width_column(csv_path: str, column_name: str) -> List[float]:
    """
    Read a specific column from widths.csv file.
    
    Args:
        csv_path: Path to the CSV file
        column_name: Name of the column to extract
        
    Returns:
        List of values from the specified column
    """
    values = []
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Check if column exists
            if column_name not in reader.fieldnames:
                print(f"⚠️  Warning: Column '{column_name}' not found in {csv_path}")
                return []
            
            for row in reader:
                try:
                    value = float(row[column_name])
                    values.append(value)
                except (ValueError, KeyError):
                    # Skip invalid values
                    continue
                    
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return []
    
    return values


def write_aggregate_csv(output_path: str, data_dict: Dict[str, List[float]]) -> None:
    """
    Write aggregated data to CSV file.
    
    Args:
        output_path: Path to output CSV file
        data_dict: Dictionary mapping column names to lists of values
    """
    if not data_dict:
        print(f"⚠️  No data to write to {output_path}")
        return
    
    # Find the maximum number of rows needed
    max_rows = max(len(values) for values in data_dict.values())
    
    # Prepare data matrix (pad shorter columns with empty strings)
    headers = sorted(data_dict.keys())
    rows = []
    
    for i in range(max_rows):
        row = []
        for header in headers:
            values = data_dict[header]
            if i < len(values):
                row.append(f"{values[i]:.3f}")
            else:
                row.append("")
        rows.append(row)
    
    # Write to CSV
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"✅ Wrote {output_path} ({len(headers)} columns, {max_rows} rows)")
        
    except Exception as e:
        print(f"❌ Error writing {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate width measurements from multiple subdirectories"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.path.expanduser("~/Datasets/fiber_new"),
        help="Root directory containing subdirectories with widths.csv files (default: ~/Datasets/fiber_new)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for summary CSV files (default: same as root-dir)"
    )
    
    args = parser.parse_args()
    
    root_dir = os.path.expanduser(args.root_dir)
    output_dir = args.output_dir if args.output_dir else root_dir
    output_dir = os.path.expanduser(output_dir)
    
    print("=" * 80)
    print("Width Measurement Aggregation")
    print("=" * 80)
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all widths.csv files
    csv_files = find_width_csv_files(root_dir)
    
    if not csv_files:
        print("❌ No widths.csv files found!")
        return
    
    print(f"Found {len(csv_files)} subdirectories with widths.csv")
    print()
    
    # Read max_width_px and max_width_scaled for each subdirectory
    max_width_px_data = {}
    max_width_scaled_data = {}
    
    for subdir_name, csv_path in csv_files:
        print(f"Processing: {subdir_name}")
        
        # Read max_width_px
        px_values = read_width_column(csv_path, "max_width_px")
        if px_values:
            max_width_px_data[subdir_name] = px_values
            print(f"  max_width_px: {len(px_values)} measurements")
        
        # Read max_width_scaled
        scaled_values = read_width_column(csv_path, "max_width_scaled")
        if scaled_values:
            max_width_scaled_data[subdir_name] = scaled_values
            print(f"  max_width_scaled: {len(scaled_values)} measurements")
        
        print()
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Write aggregate CSV files
    print("=" * 80)
    print("Writing aggregate CSV files...")
    print("=" * 80)
    
    px_output_path = os.path.join(output_dir, "max_width_px_summary.csv")
    write_aggregate_csv(px_output_path, max_width_px_data)
    
    scaled_output_path = os.path.join(output_dir, "max_width_scaled_summary.csv")
    write_aggregate_csv(scaled_output_path, max_width_scaled_data)
    
    print()
    print("=" * 80)
    print("✅ Aggregation Complete!")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - {px_output_path}")
    print(f"  - {scaled_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

