#!/bin/bash

PROJECT_PATH="$SOURCE_CODE_PATH/CausalDino"
VAL_DATA_PATH="$INPUT_PATH/UCF101"
DATA_PATH="$INPUT_PATH/videos_256"
EXP_NAME="svt_small_epic_ord128_nirvana"
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
  --use_wandb True \
  --loss TimeEmbLoss \
  --local_crops_number 0 \
  --n_global_views 4 \
  --global_crops_scale 0.14 1 \
  --loss NextTokenLoss \
  --dataset KineticsEvents \
  --local_crops_number 0 \
  --n_global_views 16 \
  --sampling_rate 2048 \
  --global_size 128 \
  --freeze_last_layer 1 \
  --predictor GPT \
  --headproba HeadProba \
  --skip_last True \
  --random_sampling False \
  --dataset_level 3 \
  --wrapper MultiCropWrapperGPT \
  --return_prediction_logits False \
  --CE_fe_c 0.5 \
  --CE_ef_c 0.5 \
  --video_extension MP4 \
  --pseudo_length 239789
