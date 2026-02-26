#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python main.py --algorithm IPPO --mode grpo --seed 42 --env_name MultiGrid-Cluttered-Fixed-15x15 --wandb_project multigrid-ippo --save_path /checkpoints/cluttered_new/grpo_return --use_return&

python main.py --algorithm IPPO --mode grpo --seed 42 --env_name MultiGrid-Cluttered-Fixed-15x15 --wandb_project multigrid-ippo --save_path /checkpoints/cluttered_new/grpo_global_norm_return --global_norm --use_return &

python main.py --algorithm IPPO --mode grpo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/meetup_new/grpo_return --use_return&

python main.py --algorithm IPPO --mode grpo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/meetup_new/grpo_global_norm_return --global_norm --use_return &

# python main.py --algorithm IPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/cluttered/IPPO &
# python main.py --algorithm MAPPO --mode ppo --seed 42 --env_name MultiGrid-Meetup-Empty-15x15-v0 --wandb_project multigrid-ippo --save_path /checkpoints/cluttered/MAPPO &

