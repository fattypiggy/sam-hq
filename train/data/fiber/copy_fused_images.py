#!/usr/bin/env python3
"""
Script to find all fused.jpg files in subdirectories and copy them to a data directory
with renamed filenames based on their parent directory names.
"""

import os
import shutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

def find_fused_jpg_files(root_dir):
    """
    Find all fused.jpg files in the directory tree.
    Returns a list of tuples: (file_path, parent_dir_name)
    """
    fused_files = []
    root_path = Path(root_dir)
    
    print(f"Searching for fused.jpg files in: {root_path}")
    
    # Use glob to find all fused.jpg files recursively
    for fused_file in root_path.rglob("fused.jpg"):
        parent_dir = fused_file.parent.name
        fused_files.append((fused_file, parent_dir))
    
    print(f"Found {len(fused_files)} fused.jpg files")
    return fused_files

def ensure_unique_names(fused_files):
    """
    Ensure all parent directory names are unique by adding suffixes if needed.
    Returns a dictionary mapping original names to unique names.
    """
    name_counts = defaultdict(int)
    unique_names = {}
    
    for _, parent_name in fused_files:
        name_counts[parent_name] += 1
    
    # Create unique names for duplicates
    for parent_name, count in name_counts.items():
        if count == 1:
            unique_names[parent_name] = parent_name
        else:
            # Add suffix for duplicates
            for i in range(count):
                if i == 0:
                    unique_names[f"{parent_name}"] = parent_name
                else:
                    unique_names[f"{parent_name}_{i+1}"] = f"{parent_name}_{i+1}"
    
    return unique_names

def copy_file(args):
    """
    Copy a single fused.jpg file to the data directory with renamed filename.
    Returns success status and file info.
    """
    fused_file_path, parent_name, data_dir, unique_name = args
    
    try:
        # Create new filename: parent_name.jpg
        new_filename = f"{unique_name}.jpg"
        dest_path = data_dir / new_filename
        
        # Copy the file
        shutil.copy2(fused_file_path, dest_path)
        
        return {
            'success': True,
            'source': str(fused_file_path),
            'destination': str(dest_path),
            'parent_name': parent_name
        }
    except Exception as e:
        return {
            'success': False,
            'source': str(fused_file_path),
            'error': str(e),
            'parent_name': parent_name
        }

def main():
    # Configuration
    root_directory = "/home/wei/Downloads/USDA"
    data_directory = "/home/wei/Downloads/USDA/data"
    max_workers = 8  # Number of threads for parallel processing
    
    print("=" * 60)
    print("Fused.jpg File Copier with Multithreading")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create data directory if it doesn't exist
    data_path = Path(data_directory)
    data_path.mkdir(exist_ok=True)
    print(f"Data directory: {data_path}")
    
    # Find all fused.jpg files
    fused_files = find_fused_jpg_files(root_directory)
    
    if not fused_files:
        print("No fused.jpg files found!")
        return
    
    # Ensure unique names for parent directories
    print("\nProcessing parent directory names...")
    unique_names = ensure_unique_names(fused_files)
    
    # Prepare arguments for multithreaded copying
    copy_args = []
    name_mapping = {}
    
    for i, (fused_file_path, parent_name) in enumerate(fused_files):
        # Get unique name for this parent directory
        if parent_name not in name_mapping:
            name_mapping[parent_name] = unique_names[parent_name]
        
        unique_name = name_mapping[parent_name]
        copy_args.append((fused_file_path, parent_name, data_path, unique_name))
    
    print(f"\nStarting file copying with {max_workers} threads...")
    
    # Track results
    successful_copies = 0
    failed_copies = 0
    results = []
    
    # Use ThreadPoolExecutor for parallel file copying
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all copy tasks
        future_to_args = {executor.submit(copy_file, args): args for args in copy_args}
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            result = future.result()
            results.append(result)
            
            if result['success']:
                successful_copies += 1
                print(f"✓ Copied: {result['parent_name']}.jpg")
            else:
                failed_copies += 1
                print(f"✗ Failed: {result['parent_name']} - {result['error']}")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files found: {len(fused_files)}")
    print(f"Successfully copied: {successful_copies}")
    print(f"Failed copies: {failed_copies}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Data directory: {data_directory}")
    
    if successful_copies > 0:
        print(f"\n✓ Successfully moved {successful_copies} images to the data directory!")
    
    if failed_copies > 0:
        print(f"\n✗ {failed_copies} files failed to copy. Check the error messages above.")

if __name__ == "__main__":
    main()
