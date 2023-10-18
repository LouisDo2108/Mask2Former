#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=0 # Change according to GPU availability

eval "$(conda shell.bash hook)"
conda activate mask2former
cd /home/dtpthao/workspace/camo/Mask2Former/

config_file="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_cod10k.yaml"
output_dir="output/vicregl_cod10k"
model_weights="$output_dir/model_final.pth"

# Train OSFormer
python train_net.py \
--config-file $config_file \
--eval-only \
MODEL.WEIGHTS $model_weights \
OUTPUT_DIR $output_dir