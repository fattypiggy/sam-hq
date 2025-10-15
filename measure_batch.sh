#!/bin/bash

# Batch processing script for measuring fiber widths across multiple datasets
# Usage: ./measure_batch.sh

ROOT_DIR="$HOME/Datasets/fiber_new"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Batch Fiber Width Measurement"
echo "========================================"
echo "Root directory: $ROOT_DIR"
echo ""

# Count total directories
total_dirs=$(find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current=0
success=0
failed=0

echo "Found $total_dirs directories to process"
echo ""

# Log file
log_file="$SCRIPT_DIR/measure_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $log_file"
echo ""

# Process each subdirectory
for BASE_DIR in "$ROOT_DIR"/*/ ; do
    # Remove trailing slash
    BASE_DIR="${BASE_DIR%/}"
    dir_name=$(basename "$BASE_DIR")
    
    current=$((current + 1))
    
    echo "----------------------------------------"
    echo "[$current/$total_dirs] Processing: $dir_name"
    echo "----------------------------------------"
    
    # Check if binary_mask_instance exists
    if [ ! -d "$BASE_DIR/binary_mask_instance" ]; then
        echo "⚠️  SKIP: No binary_mask_instance directory found"
        echo "[$current/$total_dirs] SKIP: $dir_name - No binary_mask_instance directory" >> "$log_file"
        failed=$((failed + 1))
        echo ""
        continue
    fi
    
    # Check if there are any image files
    file_count=$(find "$BASE_DIR/binary_mask_instance" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
    if [ "$file_count" -eq 0 ]; then
        echo "⚠️  SKIP: No image files found in binary_mask_instance"
        echo "[$current/$total_dirs] SKIP: $dir_name - No image files" >> "$log_file"
        failed=$((failed + 1))
        echo ""
        continue
    fi
    
    echo "   Input:  $BASE_DIR/binary_mask_instance ($file_count files)"
    echo "   Output: $BASE_DIR/widths.csv"
    echo ""
    
    # Run the measurement script
    start_time=$(date +%s)
    
    python tools/measure_binary_instance_widths.py \
        --input-dir "$BASE_DIR/binary_mask_instance" \
        --output-dir "$BASE_DIR" \
        --original-image-dir "$BASE_DIR" \
        --csv-name widths.csv \
        --binary-thresh 127 \
        --sample-stride 2 \
        --pca-radius 21 \
        --cast-step 0.5 \
        --max-cast-steps 4096 \
        --save-skeleton \
        --skeleton-radius 1 \
        --prune \
        --prune-min-length 10.0 \
        --max-width-thresh 100.0 \
        --iqr-multiplier 1.5 \
        --gradient-threshold 2.0 \
        --max-skeleton-jump 2.0
    
    exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS - Completed in ${elapsed}s"
        echo "[$current/$total_dirs] SUCCESS: $dir_name (${elapsed}s)" >> "$log_file"
        success=$((success + 1))
    else
        echo "❌ FAILED - Exit code: $exit_code"
        echo "[$current/$total_dirs] FAILED: $dir_name - Exit code: $exit_code" >> "$log_file"
        failed=$((failed + 1))
    fi
    
    echo ""
done

echo "========================================"
echo "Batch Processing Complete"
echo "========================================"
echo "Total directories: $total_dirs"
echo "Successful: $success"
echo "Failed/Skipped: $failed"
echo "Log file: $log_file"
echo "========================================"

