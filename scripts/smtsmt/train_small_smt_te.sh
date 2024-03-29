#!/bin/bash

PROJECT_PATH="$HOME/CausalDino"
DATA_PATH="/mnt/data/something-something-v2/raw_mp4"
EXP_NAME="svt_small_smt_te"
PORT='1033'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir -p "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 16 \
  --data_path "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
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
  --dataset KineticsEvents \
  --wrapper MultiCropWrapperGPT \
  --return_prediction_logits False \
  --predictor GPT2FoldPredictor \
  --headproba HeadProba \
  --skip_last True \
  --random_sampling False \
  --CE_fe_c 0.5 \
  --CE_ef_c 0.5 \
  --video_extension mp4

