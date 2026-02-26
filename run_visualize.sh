#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT_BASE="/home/brianko/test/multigrid/checkpoints"
ENV="MultiGrid-Cluttered-Fixed-15x15"

# Visualize IPPO
# python main.py \
#     --visualize \
#     --algorithm IPPO \
#     --mode ppo \
#     --seed 42 \
#     --env_name $ENV \
#     --load_checkpoint_from $CHECKPOINT_BASE/IPPO/run_1771842788/ep_30000 \
#     --video_dir videos/IPPO

# # Visualize MAPPO
# python main.py \
#     --visualize \
#     --algorithm MAPPO \
#     --mode ppo \
#     --seed 42 \
#     --env_name $ENV \
#     --load_checkpoint_from $CHECKPOINT_BASE/MAPPO/run_1771842788/ep_30000 \
#     --video_dir videos/MAPPO

# Visualize IGRPO
python main.py \
    --visualize \
    --algorithm IPPO \
    --mode grpo \
    --seed 42 \
    --env_name $ENV \
    --global_norm \
    --load_checkpoint_from $CHECKPOINT_BASE/cluttered_new/grpo_global_norm_return/run_1772078651/ep_30000 \
    --video_dir videos/IGRPO
