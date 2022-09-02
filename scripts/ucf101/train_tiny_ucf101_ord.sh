
PROJECT_PATH="$HOME/CausalDino"
DATA_PATH="/mnt/data/UCF101"
EXP_NAME="tiny_ucf101_ord"
PORT='1029'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE="run"
export WANDB_API_KEY="df61f407e5d9259d358ba2a7ef24aa3038bec740"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$PORT" \
  train_ssl.py \
  --arch "timesformer" \
  --model_name get_deit_tiny_patch16_224 \
  --batch_size_per_gpu 32 \
  --data_path "${DATA_PATH}" \
  --output_dir "$PROJECT_PATH/checkpoints/$EXP_NAME" \
  --exp_name $EXP_NAME \
  --do_eval True \
  --eval_freq 4 \
  --use_wandb True \
  --loss FeatureLoss \
  --dataset KineticsEvents \
  --local_crops_number 0 \
  --n_global_views 4 \
  --n_parts 5 \
  --freeze_last_layer 1 \
  --global_crops_scale 0.14 1 \
  --wrapper MultiCropWrapperGPT \
  --predictor GPT \
  --predictor_model_type gpt-micro-256-half \
  --headproba HeadProba \
  --skip_last True
#  --lr 3e-4 \
#  --clip_grad 1.0 