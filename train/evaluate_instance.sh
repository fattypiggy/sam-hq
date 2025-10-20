#!/bin/bash
# Evaluation script for comparing three models on instance-level segmentation
# This script evaluates on 40 validation images from data/fiber_split/val
# Metrics: IoU and clDice
#
# Models compared:
# 1. Vanilla SAM - Original SAM without HQ decoder
# 2. SAM-HQ Baseline - Pretrained SAM-HQ model
# 3. SAM-HQ Trained - Our trained model with skeleton loss

# Configuration
MODEL_TYPE="vit_l"
SAM_CHECKPOINT="./pretrained_checkpoint/sam_vit_l_0b3195.pth"
SAM_HQ_BASELINE="./pretrained_checkpoint/sam_hq_vit_l.pth"
SAM_HQ_TRAINED="./work_dirs/train_hq_sam_l_epoch200_instance_200images_skeleton-loss-weight1.0/epoch_52.pth"
OUTPUT_BASE="./work_dirs/eval_comparison_instance"

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
echo "Instance-Level Evaluation Comparison"
echo "======================================"
echo "Model Type: $MODEL_TYPE"
echo "Validation Set: data/fiber_split/val (40 images)"
echo "Metrics: IoU, clDice, Boundary IoU"
echo ""

# 1. Evaluate vanilla SAM (no HQ decoder)
echo "----------------------------------------"
echo "1/3 Evaluating Vanilla SAM..."
echo "----------------------------------------"
torchrun --nproc_per_node=1 ./train.py \
  --eval \
  --checkpoint "$SAM_CHECKPOINT" \
  --model-type "$MODEL_TYPE" \
  --output "${OUTPUT_BASE}/vanilla_sam" \
  --vis-branch vanilla \
  --visualize \
  --instance

echo ""
echo "✓ Vanilla SAM evaluation complete"
echo "Results saved to: ${OUTPUT_BASE}/vanilla_sam"
echo ""

# 2. Evaluate SAM-HQ baseline (pretrained HQ decoder)
echo "----------------------------------------"
echo "2/3 Evaluating SAM-HQ Baseline..."
echo "----------------------------------------"
torchrun --nproc_per_node=1 ./train.py \
  --eval \
  --checkpoint "$SAM_CHECKPOINT" \
  --model-type "$MODEL_TYPE" \
  --restore-model "$SAM_HQ_BASELINE" \
  --output "${OUTPUT_BASE}/sam_hq_baseline" \
  --vis-branch hq \
  --visualize \
  --instance

echo ""
echo "✓ SAM-HQ Baseline evaluation complete"
echo "Results saved to: ${OUTPUT_BASE}/sam_hq_baseline"
echo ""

# 3. Evaluate SAM-HQ with our trained weights
echo "----------------------------------------"
echo "3/3 Evaluating SAM-HQ Trained (Ours)..."
echo "----------------------------------------"
torchrun --nproc_per_node=1 ./train.py \
  --eval \
  --checkpoint "$SAM_CHECKPOINT" \
  --model-type "$MODEL_TYPE" \
  --restore-model "$SAM_HQ_TRAINED" \
  --output "${OUTPUT_BASE}/sam_hq_trained" \
  --vis-branch hq \
  --visualize \
  --instance

echo ""
echo "✓ SAM-HQ Trained evaluation complete"
echo "Results saved to: ${OUTPUT_BASE}/sam_hq_trained"
echo ""

# 4. Generate comparison summary
echo "======================================"
echo "Generating Comparison Summary..."
echo "======================================"
python summarize_metrics.py --eval-dir "$OUTPUT_BASE" --num-images 40
