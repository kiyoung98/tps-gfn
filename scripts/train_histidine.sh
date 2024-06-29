current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project histidine \
    --molecule histidine \
    --date $current_date \
    --seed $seed \
    --wandb \
    --save_freq 1 \
    --sigma 0.1 \
    --num_steps 5000 \
    --buffer_size 1024 \
    --num_rollouts 1000 \
    --trains_per_rollout 1000 &
done

wait