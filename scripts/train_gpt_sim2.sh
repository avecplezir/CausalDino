#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
#DATA_PATH="$HOME/kinetics-dataset/k400/videos_train_256p_dense_cache"
DATA_PATH="/mnt/data/UCF101"
EXP_NAME="svt_ucf101_gpt_sim2"
PORT='1026'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0
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
  --do_eval True \
  --use_wandb True \
  --loss GPTSim2Loss \
  --dataset KineticsEvents \
  --local_crops_number 0 \
  --n_global_views 4 \
  --freeze_last_layer 1 \
  --global_crops_scale 0.14 1 \
  --predictor GPT \
  --n_parts 9 \
  --skip_last True
