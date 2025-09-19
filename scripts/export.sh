python /mnt/c/GitHub/sam-hq/scripts/export_hqsam_split_onnx.py \
  --model-type vit_b \
  --sam-checkpoint /mnt/c/GitHub/sam-hq/train/pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --hq-decoder-checkpoint /mnt/c/GitHub/sam-hq/train/work_dirs/train_hq_sam_b_100_instance-200images/epoch_38.pth \
  --encoder-out /mnt/c/GitHub/sam-hq/train/onnx/encoder_vit_b.onnx \
  --decoder-out /mnt/c/GitHub/sam-hq/train/onnx/decoder_vit_b.onnx \
  --opset 17 \
  --multimask-output