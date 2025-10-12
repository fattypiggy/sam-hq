#!/bin/bash

# 设置基础目录
BASE_DIR=~/Datasets/fiber

# 遍历所有子文件夹
for dir in "$BASE_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        echo "================================================"
        echo "Processing directory: $dir"
        echo "================================================"

        python demo/demo_hqsam_auto_instances.py \
        --checkpoint train/pretrained_checkpoint/sam_vit_l_0b3195.pth \
        --restore-model train/ckpts/vit_l_loss.pth \
        --model-type vit_l \
        --input "$dir" \
        --output "$dir" \
        --device cuda \
        --points-per-side 64 \
        --pred-iou-thresh 0.6 \
        --stability-score-thresh 0.925 \
        --min-hole-region-area 100 \
        --min-island-region-area 3000 \
        --min-instance-area 2000 \
        --max-instance-area-ratio 0.1 \
        --overlap-thresh 0.2 \
        --resume-index 0

        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: $dir"
        else
            echo "✗ Error processing: $dir"
        fi
        echo ""
    fi
done

echo "All directories processed!"
