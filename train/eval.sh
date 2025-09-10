torchrun --nproc_per_node=1 ./train.py \
  --eval \
  --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --model-type vit_l \
  --restore-model ./work_dirs/train_hq_sam_l_100_instance/epoch_99.pth \
  --output ./work_dirs/eval_hq_sam_l_100_instance_2 \
  --vis-branch hq \
  --visualize \
  --instance