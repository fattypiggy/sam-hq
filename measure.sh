BASE_DIR="$HOME/Datasets/fiber_new/A01-029 1-1"
python tools/measure_binary_instance_widths.py \
--input-dir "$BASE_DIR/binary_mask_instance" \
--output-dir $BASE_DIR \
--original-image-dir $BASE_DIR \
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