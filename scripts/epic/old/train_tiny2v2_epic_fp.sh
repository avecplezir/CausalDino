#!/bin/bash

SOURCE_CODE_PATH=$HOME
PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
SNAPSHOT_PATH="$PROJECT_PATH/checkpoints"
VAL_DATA_PATH="/mnt/data/UCF101"
DATA_PATH="/mnt/data/EPIC-KITCHENS-100/videos_256"
PORT='1025'

EXP_NAME="tiny2v2_epic_fp"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

export CUDA_VISIBLE_DEVICES=4

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
  --eval_freq 1 \
  --weight_decay_end 0.1 \
  --n_global_views 2 \
  --local_crops_number 0 \
  --global_crops_scale 0.14 1 \
  --n_parts 11 \
  --use_wandb True \
  --loss FeatureLossAllPairs \
  --dataset EpicEvents \
  --wrapper MultiCropWrapperPredictorProjector \
  --predictor MLPfeaturePredictor \
  --headproba HeadProba \
  --skip_last True \
  --CE_fe_c 1. \
  --CE_ef_c 0. \
  --return_pred_out True \
  --do_eval_before_train True \