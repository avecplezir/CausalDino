#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
#DATA_PATH="$HOME/kinetics-dataset/k400/videos_train_256p_dense_cache"
#DATA_PATH="/mnt/data/UCF101"
#DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/something-something-v2"
EXP_NAME="svt_ucf101_nirvana"
#DATA_PATH="/mnt/data/Kinetics/videos_val_256p_dense_cache"
#EXP_NAME="svt_k400"
PORT='1024'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 32 \
  --data_path "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
  --exp_name $EXP_NAME \
  --model_name get_vit_base_patch16_224 \
  --do_eval True \
  --eval_freq 2 \
  --n_global_views 2 \
  --n_parts 11 \
  --use_wandb True \
  --loss DINOLoss \
  --dataset Kinetics \
  --video_extension mp4
