OUTPUT_DIR=work_dirs/train_hq_sam_b_epoch200_instance_200images_baseline
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=1 ./train.py \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --max_epoch_num 200 \
  --lr_drop_epoch 25 \
  --output "$OUTPUT_DIR" \
  --instance \
  --use-skeleton-loss \
  --skeleton-loss-weight 1.0 \
  --skeleton-tube-radius 2 \
  2>&1 | tee -a "$OUTPUT_DIR/console.log"
