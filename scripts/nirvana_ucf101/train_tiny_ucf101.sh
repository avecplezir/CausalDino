#!/bin/bash

PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
DATA_PATH="$INPUT_PATH/UCF101"
EXP_NAME="ucf101_tiny_nirvana"
PORT='1024'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 32 \
  --data_path "${DATA_PATH}" \
  --val_data_dir "${DATA_PATH}" \
  --output_dir "${SNAPSHOT_PATH}/${EXP_NAME}" \
  --exp_name $EXP_NAME \
  --model_name get_deit_small_patch16_224 \
  --do_eval True \
  --eval_freq 5 \
  --epochs 100 \
  --warmup_epochs 20 \
  --weight_decay_end 0.1 \
  --saveckp_freq 20 \
  --use_wandb True \
  --loss DINOLoss \
  --dataset Kinetics \
  --video_extension avi

