#!/bin/bash

PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
VAL_DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/videos_256"
EXP_NAME="tiny_epic_bn_memory64maxlen8_nirvana"
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
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  \
  --batch_size_per_gpu 64 \
  --do_eval True \
  --sampling_rate 16 \
  --eval_freq 5 \
  --use_wandb True \
  --weight_decay_end 0.1 \
  --num_workers 24 \
  \
  --dataset EpicNFEvents \
  --continuous True \
  --loss MemoryLoss \
  --local_crops_number 0 \
  --n_global_views 1 \
  --random_sampling False \
  --maxlen 8 \
  --block_size 8 \
  --n_parts 8 \
  --global_crops_scale 0.14 1 \
  --wrapper MultiCropWrapperGeneral \
  --predictor GPT \
  --head Projector \
  --headproba HeadProbal2Norm \
  --loss_mode memory \
  --CE_fe_c 1 \
  --CE_ef_c 0. \
  --use_bn_in_head True \
  --hidden_dim_in_head 2048 \
  --teacher_prediction_type head \
  --student_prediction_type head_first \
  --maskemb True \
  --layer_norm_in_head False \
  --l2norm_in_head False \
