#!/bin/bash

TERRANOVA=false

if [ "$TERRANOVA" = true ]; then
SOURCE_CODE_PATH=$HOME
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
SNAPSHOT_PATH="$PROJECT_PATH/checkpoints"
VAL_DATA_PATH="/mnt/data/UCF101"
DATA_PATH="/mnt/data/EPIC-KITCHENS-100/videos_256"
export CUDA_VISIBLE_DEVICES=3
fi

if [ "$TERRANOVA" = false ]; then
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
VAL_DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/videos_256"
fi


EXP_NAME="tiny_epic_s1dp_repeat_bn"
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
  --data_path "${DATA_PATH}" \
  --val_data_dir "${VAL_DATA_PATH}" \
  --output_dir "${SNAPSHOT_PATH}/${EXP_NAME}" \
  --exp_name $EXP_NAME \
  --video_extension MP4 \
  --dataset_level 3 \
  \
  --arch "timesformer" \
  --batch_size_per_gpu 32 \
  --model_name get_deit_tiny_patch16_224 \
  \
  --do_eval True \
  --eval_freq 5 \
  --weight_decay_end 0.1 \
  --n_global_views 1 \
  --local_crops_number 0 \
  --global_crops_scale 0.14 1 \
  --n_parts 11 \
  --use_wandb True \
  --wrapper MultiCropWrapperDinoMemory \
  --head Projector \
  --headproba HeadProbal2NormDp \
  --memory PatchMemory \
  --loss DinoMemoryLoss \
  --random_sampling False \
  --maxlen 7 \
  --block_size 7 \
  --n_parts 7 \
  --dataset EpicEvents \
  --continuous_repeat True \
  --use_bn_in_head True \
  --num_workers 30 \
  --teacher_prediction_type head \
  --student_prediction_type head_first \
  --temporal_aug True \


