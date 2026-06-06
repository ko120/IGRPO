#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Unified experiment matrix: 2 envs x 4 configs x 3 seeds = 24 runs.
# For each (env, config) the 3 seeds launch in parallel; `wait`
# between config batches caps peak GPU concurrency at 3. Each run
# writes to its own (env, config, seed) save_path so parallel runs
# never collide. To add an environment, append its id to ENVS.
# ============================================================
ENVS="MultiGrid-Cluttered-Fixed-15x15 MultiGrid-FourRooms-15x15-v0"
SEEDS="42 1 7"
WB="multigrid-ippo"

for env in $ENVS; do
    for s in $SEEDS; do
        python main.py --algorithm IPPO --mode ppo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/IPPO/seed_$s" &
    done
    wait

    for s in $SEEDS; do
        python main.py --algorithm MAPPO --mode ppo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/MAPPO/seed_$s" &
    done
    wait

    for s in $SEEDS; do
        python main.py --algorithm IPPO --mode grpo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/grpo_return/seed_$s" --use_return &
    done
    wait

    for s in $SEEDS; do
        python main.py --algorithm IPPO --mode grpo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/grpo_global_norm_return/seed_$s" --global_norm --use_return &
    done
    wait
done
