#!/bin/bash

SOURCE_CODE_PATH=$HOME
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
SNAPSHOT_PATH="$PROJECT_PATH/checkpoints"
VAL_DATA_PATH="/mnt/data/UCF101"
DATA_PATH="/mnt/data/EPIC-KITCHENS-100/videos_256"
PORT='1024'

EXP_NAME="tiny_epic_memory_loc_ta2"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --data_path "${DATA_PATH}" \
  --val_data_dir "${VAL_DATA_PATH}" \
  --output_dir "${SNAPSHOT_PATH}/${EXP_NAME}" \
  --video_extension MP4 \
  --dataset_level 3 \
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  --batch_size_per_gpu 32 \
  --exp_name $EXP_NAME \
  --do_eval True \
  --eval_freq 5 \
  --use_wandb False \
  --loss MemoryLoss \
  --maxlen 8 \
  --block_size 8 \
  --CE_fe_c 1. \
  --CE_ef_c 0.5 \
  --CE_ee_c 0.5 \
  --dataset EpicNFEvents \
  --continuous True \
  --local_crops_number 8 \
  --n_global_views 2 \
  --freeze_last_layer 1 \
  --global_crops_scale 0.4 1 \
  --weight_decay_end 0.1 \
  --wrapper MultiCropWrapperMemory \
  --predictor GPT \
  --random_sampling False \
