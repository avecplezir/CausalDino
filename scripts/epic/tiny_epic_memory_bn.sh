#!/bin/bash

SOURCE_CODE_PATH=$HOME
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
SNAPSHOT_PATH="$PROJECT_PATH/checkpoints"
VAL_DATA_PATH="/mnt/data/UCF101"
DATA_PATH="/mnt/data/EPIC-KITCHENS-100/videos_256"
PORT='1027'

EXP_NAME="tiny_epic_memory3_bn"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

export CUDA_VISIBLE_DEVICES=1

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
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  \
  --batch_size_per_gpu 4 \
  --do_eval True \
  --eval_freq 5 \
  --use_wandb True \
  --weight_decay_end 0.1 \
  --num_workers 10 \
  \
  --dataset EpicNFEvents \
  --continuous True \
  --loss GPTLoss \
  --local_crops_number 0 \
  --n_global_views 4 \
  --random_sampling False \
  --maxlen 16 \
  --block_size 64 \
  --n_parts 64 \
  --global_crops_scale 0.14 1 \
  --wrapper MultiCropWrapperGeneral \
  --predictor GPT \
  --head Projector \
  --headproba HeadProbal2Norm \
  --memory PatchMemory \
  --loss_mode memory_gpt \
  --CE_fe_c 1 \
  --CE_ef_c 0. \
  --use_bn_in_head True \
  --hidden_dim_in_head 2048 \
  --teacher_prediction_type head \
  --student_prediction_type head_first \
  --lr 1e-3 \
  --min_lr 1e-5 \



