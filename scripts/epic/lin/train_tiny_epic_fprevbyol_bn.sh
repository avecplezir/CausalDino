#!/bin/bash

SOURCE_CODE_PATH=$HOME
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
SNAPSHOT_PATH="$PROJECT_PATH/checkpoints"
VAL_DATA_PATH="/mnt/data/UCF101"
DATA_PATH="/mnt/data/EPIC-KITCHENS-100/videos_256"
PORT='1025'

EXP_NAME="tiny_epic_fprevbyol_bn"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

export CUDA_VISIBLE_DEVICES=2

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
  --batch_size_per_gpu 32 \
  --model_name get_deit_tiny_patch16_224 \
  --do_eval True \
  --eval_freq 5 \
  --weight_decay_end 0.1 \
  --n_global_views 2 \
  --local_crops_number 0 \
  --global_crops_scale 0.14 1 \
  --n_parts 11 \
  --num_workers 20 \
  --use_wandb True \
  --loss FeatureLossAllPairs \
  --dataset EpicEvents \
  --wrapper MultiCropWrapperGPT \
  --predictor MLPBYOL \
  --head MLPBYOL \
  --headproba HeadProbal2Norm \
  --CE_fe_c 1. \
  --CE_ef_c 0. \
  --use_bn_in_head True \
  --use_bn_in_pred True \
  --hidden_dim_in_head 4096 \
  --hidden_dim_in_pred 4096 \
  --layer_norm_in_pred False \
  --layer_norm_in_head False \
  --num_workers 20 \