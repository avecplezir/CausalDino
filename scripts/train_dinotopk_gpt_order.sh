
PROJECT_PATH="$HOME/CausalDino"
#DATA_PATH="$HOME/kinetics-dataset/k400/videos_train_256p_dense_cache"
DATA_PATH="/mnt/data/UCF101"
EXP_NAME="svt_ucf101_topk_order_32"
PORT='1033'

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

#export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0
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
  --loss DINOTopkLoss \
  --dataset KineticsEvents \
  --local_crops_number 0 \
  --n_global_views 5 \
  --random_sampling False \
  --freeze_last_layer 1 \
  --global_crops_scale 0.14 1 \
  --predictor GPT \
  --headproba HeadProba