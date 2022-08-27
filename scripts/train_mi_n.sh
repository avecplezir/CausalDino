#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
DATA_PATH="/mnt/data/UCF101"
EXP_NAME="svt_ucf101_tiny_mi_n"
PORT='1026'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --out_dim 20 \
  --batch_size_per_gpu 64 \
  --data_path "${DATA_PATH}" \
  --val_data_dir "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
  --exp_name $EXP_NAME \
  --model_name get_deit_tiny_patch16_224 \
  --yt_path //home/yr/ianokhin \
  --do_eval True \
  --eval_freq 2 \
  --n_global_views 2 \
  --local_crops_number 0 \
  --global_crops_scale 0.14 1 \
  --n_parts 11 \
  --use_wandb True \
  --loss DINOMI3Loss \
  --dataset Kinetics \
  --video_extension avi
