torchrun --nproc_per_node=1 ./train.py \
  --eval \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --restore-model ./work_dirs/train_hq_sam_b_epoch200_instance_200images_skeleton-loss-weight1.0/epoch_29.pth \
  --output ./work_dirs/eval_hq_sam_b_instance_200images_skeleton-loss-weight1.0 \
  --vis-branch hq \
  --visualize \
  --instance
