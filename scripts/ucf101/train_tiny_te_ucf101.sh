
PROJECT_PATH="$HOME/CausalDino"
DATA_PATH="/mnt/data/UCF101"
EXP_NAME="tiny_ucf101_te"
PORT='1030'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=1,2
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  --batch_size_per_gpu 16 \
  --data_path "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
  --yt_path //home/yr/ianokhin \
  --exp_name $EXP_NAME \
  --do_eval True \
  --eval_freq 1 \
  --use_wandb True \
  --loss TimeEmbLoss \
  --dataset KineticsEvents \
  --local_crops_number 0 \
  --n_global_views 5 \
  --freeze_last_layer 1 \
  --global_crops_scale 0.14 1 \
  --wrapper MultiCropWrapperTimeEmb \
  --predictor GPT2FoldPredictor \
  --headproba HeadProba \
  --skip_last True \
  --random_sampling False \
  --CE_fe_c 0.5 \
  --CE_ef_c 0.5 \
  --video_extension avi

