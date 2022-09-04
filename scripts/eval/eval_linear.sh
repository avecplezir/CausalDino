#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
EXP_NAME="le_001"
DATASET="ucf101"
DATA_PATH="/mnt/data/ucf101"
CHECKPOINT="/home/ivananokhin/.cache/torch/hub/checkpoints/kinetics400_vitb_ssl.pth"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_linear.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 8e-3 \
  --batch_size_per_gpu 32 \
  --num_workers 4 \
  --num_labels 101 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}" \
  DATA.PATH_PREFIX "$DATA_PATH/videos_256p_dense_cache" \
  DATA.USE_FLOW False
