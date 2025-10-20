#!/bin/bash
# Test script for comparing three models on automatic instance segmentation
# Dataset: STD-04 1-1
# Mode: Automatic mask generation (no prompts)
#
# Models compared:
# 1. Vanilla SAM - Original SAM without HQ decoder
# 2. SAM-HQ Baseline - Pretrained SAM-HQ model
# 3. SAM-HQ Trained - Our trained model with skeleton loss

# Configuration
MODEL_TYPE="vit_l"
SAM_CHECKPOINT="train/pretrained_checkpoint/sam_vit_l_0b3195.pth"
SAM_HQ_BASELINE="train/pretrained_checkpoint/sam_hq_vit_l.pth"
SAM_HQ_TRAINED="train/work_dirs/train_hq_sam_l_epoch200_instance_200images_skeleton-loss-weight1.0/epoch_52.pth"

# Input/Output paths
INPUT_DIR="$HOME/Datasets/fiber_new/STD-04 1-1"
OUTPUT_BASE="test_results/STD-04_1-1_auto_comparison"

# Automatic mask generation parameters (matching original test.sh)
POINTS_PER_SIDE=64
PRED_IOU_THRESH=0.6
STABILITY_SCORE_THRESH=0.925
MIN_HOLE_REGION_AREA=100
MIN_ISLAND_REGION_AREA=3000
MIN_INSTANCE_AREA=2000
MAX_INSTANCE_AREA_RATIO=0.1
OVERLAP_THRESH=0.2
RESUME_INDEX=0

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Verify checkpoints exist
echo "======================================"
echo "Verifying checkpoints..."
echo "======================================"

if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "❌ ERROR: SAM checkpoint not found: $SAM_CHECKPOINT"
    exit 1
fi
echo "✓ SAM checkpoint: $SAM_CHECKPOINT"

if [ ! -f "$SAM_HQ_BASELINE" ]; then
    echo "❌ ERROR: SAM-HQ baseline not found: $SAM_HQ_BASELINE"
    exit 1
fi
echo "✓ SAM-HQ baseline: $SAM_HQ_BASELINE"

if [ ! -f "$SAM_HQ_TRAINED" ]; then
    echo "❌ ERROR: SAM-HQ trained model not found: $SAM_HQ_TRAINED"
    exit 1
fi
echo "✓ SAM-HQ trained: $SAM_HQ_TRAINED"

echo ""
echo "======================================"
echo "Automatic Instance Segmentation Test"
echo "======================================"
echo "Model Type: $MODEL_TYPE"
echo "Input: $INPUT_DIR"
echo "Output Base: $OUTPUT_BASE"
echo "Mode: Automatic (no prompts)"
echo "Points per side: $POINTS_PER_SIDE"
echo ""

# 1. Test vanilla SAM (no HQ decoder)
echo "----------------------------------------"
echo "1/3 Testing Vanilla SAM (first 10 images)..."
echo "----------------------------------------"
python demo/demo_hqsam_auto_instances.py \
  --checkpoint "$SAM_CHECKPOINT" \
  --model-type "$MODEL_TYPE" \
  --input "$INPUT_DIR" \
  --output "${OUTPUT_BASE}/vanilla_sam" \
  --device cuda \
  --points-per-side $POINTS_PER_SIDE \
  --pred-iou-thresh $PRED_IOU_THRESH \
  --stability-score-thresh $STABILITY_SCORE_THRESH \
  --min-hole-region-area $MIN_HOLE_REGION_AREA \
  --min-island-region-area $MIN_ISLAND_REGION_AREA \
  --min-instance-area $MIN_INSTANCE_AREA \
  --max-instance-area-ratio $MAX_INSTANCE_AREA_RATIO \
  --overlap-thresh $OVERLAP_THRESH \
  --resume-index $RESUME_INDEX \
  --max-images 10

echo ""
echo "✓ Vanilla SAM test complete"
echo "Results: ${OUTPUT_BASE}/vanilla_sam"
echo ""

# 2. Test SAM-HQ baseline (pretrained HQ decoder)
echo "----------------------------------------"
echo "2/3 Testing SAM-HQ Baseline (first 10 images)..."
echo "----------------------------------------"
python demo/demo_hqsam_auto_instances.py \
  --checkpoint "$SAM_CHECKPOINT" \
  --restore-model "$SAM_HQ_BASELINE" \
  --model-type "$MODEL_TYPE" \
  --input "$INPUT_DIR" \
  --output "${OUTPUT_BASE}/sam_hq_baseline" \
  --device cuda \
  --points-per-side $POINTS_PER_SIDE \
  --pred-iou-thresh $PRED_IOU_THRESH \
  --stability-score-thresh $STABILITY_SCORE_THRESH \
  --min-hole-region-area $MIN_HOLE_REGION_AREA \
  --min-island-region-area $MIN_ISLAND_REGION_AREA \
  --min-instance-area $MIN_INSTANCE_AREA \
  --max-instance-area-ratio $MAX_INSTANCE_AREA_RATIO \
  --overlap-thresh $OVERLAP_THRESH \
  --resume-index $RESUME_INDEX \
  --max-images 10

echo ""
echo "✓ SAM-HQ Baseline test complete"
echo "Results: ${OUTPUT_BASE}/sam_hq_baseline"
echo ""

# 3. Test SAM-HQ with trained weights
echo "----------------------------------------"
echo "3/3 Testing SAM-HQ Trained (Ours, first 10 images)..."
echo "----------------------------------------"
python demo/demo_hqsam_auto_instances.py \
  --checkpoint "$SAM_CHECKPOINT" \
  --restore-model "$SAM_HQ_TRAINED" \
  --model-type "$MODEL_TYPE" \
  --input "$INPUT_DIR" \
  --output "${OUTPUT_BASE}/sam_hq_trained" \
  --device cuda \
  --points-per-side $POINTS_PER_SIDE \
  --pred-iou-thresh $PRED_IOU_THRESH \
  --stability-score-thresh $STABILITY_SCORE_THRESH \
  --min-hole-region-area $MIN_HOLE_REGION_AREA \
  --min-island-region-area $MIN_ISLAND_REGION_AREA \
  --min-instance-area $MIN_INSTANCE_AREA \
  --max-instance-area-ratio $MAX_INSTANCE_AREA_RATIO \
  --overlap-thresh $OVERLAP_THRESH \
  --resume-index $RESUME_INDEX \
  --max-images 10

echo ""
echo "✓ SAM-HQ Trained test complete"
echo "Results: ${OUTPUT_BASE}/sam_hq_trained"
echo ""

# 4. Summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "All three models have been tested on: $INPUT_DIR"
echo ""
echo "Results saved to:"
echo "  1. Vanilla SAM:     ${OUTPUT_BASE}/vanilla_sam"
echo "  2. SAM-HQ Baseline: ${OUTPUT_BASE}/sam_hq_baseline"
echo "  3. SAM-HQ Trained:  ${OUTPUT_BASE}/sam_hq_trained"
echo ""
echo "Each output directory contains:"
echo "  - masks/               : Image-level overlays (all instances)"
echo "  - masks_inst/          : Instance-level overlays"
echo "  - binary_mask_instance/: Binary masks for each instance"
echo ""
echo "You can compare the results by viewing the images in the masks/ folders."
echo "======================================"
