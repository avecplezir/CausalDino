#!/bin/bash

PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
VAL_DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/raw_mp4"
EXP_NAME="svt_small_smt_nirvana"
PORT='1024'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 16 \
  --data_path "${DATA_PATH}" \
  --val_data_dir "${VAL_DATA_PATH}" \
  --output_dir "${SNAPSHOT_PATH}/${EXP_NAME}" \
  --exp_name $EXP_NAME \
  --model_name get_deit_small_patch16_224 \
  --do_eval True \
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
  --video_extension mp4
