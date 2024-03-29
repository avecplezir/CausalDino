#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
DATA_PATH="/mnt/data/Kinetics/videos_train_256p_dense_cache"
#VAL_DATA_PATH="/mnt/data/ucf101/videos_256p_dense_cache"
#VAL_DATA_PATH="/mnt/data/hmdb51/videos_256"
VAL_DATA_PATH="/home/ivananokhin/mmaction2/data/hmdb51/videos"
EXP_NAME="base_k400_full_kinetics400_vitb_ssl"
PORT='1024'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 8 \
  --data_path "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
  --val_data_dir "${VAL_DATA_PATH}" \
  --exp_name $EXP_NAME \
  --model_name get_vit_base_patch16_224 \
  --default_cfg svt_vit_base_patch16_224 \
  --do_eval True \
  --do_eval_before_train True \
  --eval_freq 1 \
  --epochs 20 \
  --warmup_epochs 5 \
  --weight_decay_end 0.1 \
  --saveckp_freq 10 \
  --n_global_views 2 \
  --n_parts 11 \
  --use_wandb True \
  --loss DINOLoss \
  --dataset Kinetics \
  --video_extension mp4 \
  --full_pretrain ~/.cache/torch/hub/checkpoints/kinetics400_vitb_ssl.pth \
  --eval_dataset2 "" \
  --eval_dataset HMDBReturnIndexDataset \

