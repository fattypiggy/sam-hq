OUTPUT_DIR=work_dirs/train_hq_sam_l_100_instance-109images
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=1 ./train.py \
  --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --model-type vit_l \
  --max_epoch_num 200 \
  --lr_drop_epoch 20 \
  --output "$OUTPUT_DIR" \
  --instance 2>&1 | tee -a "$OUTPUT_DIR/console.log"
