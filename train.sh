#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=phoenix3
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=999:00:00

export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=0 # Change according to GPU availability
export CUDA_LAUNCH_BLOCKING=1

eval "$(conda shell.bash hook)"
conda activate mask2former
cd /home/dtpthao/workspace/camo/Mask2Former/

config_file="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_cod10k.yaml"
output_dir="output/vicregl_cod10k_timm_pretrained_adamw_100ep"
# model_weights="/home/dtpthao/workspace/camo/Mask2Former/output/vicregl_cod10k/model_0009999.pth"

# Train mask2former
python train_net.py \
--num-gpus 1 \
--config-file $config_file \
OUTPUT_DIR $output_dir
# --eval-only \