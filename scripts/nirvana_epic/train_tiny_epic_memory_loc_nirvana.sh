#!/bin/bash

PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
VAL_DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/videos_256"
EXP_NAME="tiny_epic_memory_loc_nirvana"
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
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  --batch_size_per_gpu 32 \
  --exp_name $EXP_NAME \
  --do_eval True \
  --eval_freq 5 \
  --use_wandb True \
  --loss MemoryLoss \
  --maxlen 128 \
  --block_size 128 \
  --CE_fe_c 1. \
  --CE_ef_c 0.5 \
  --CE_ee_c 0.5 \
  --dataset EpicNFEvents \
  --sampling_rate 10 \
  --num_workers 10 \
  --continuous True \
  --local_crops_number 8 \
  --n_global_views 2 \
  --freeze_last_layer 1 \
  --global_crops_scale 0.4 1 \
  --weight_decay_end 0.1 \
  --wrapper MultiCropWrapperMemory \
  --predictor GPT \
  --random_sampling False \
  --video_extension MP4 \
  --dataset_level 3 \
