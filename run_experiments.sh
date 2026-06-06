#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# Stop each process from grabbing every core (avoids thread thrash with many runs).
export OMP_NUM_THREADS=1

# ============================================================
# Experiment matrix: 2 envs x 4 configs x 3 seeds = 24 runs.
#
# Each run starts its OWN Ray instance (ray.init per process). Launching all 24
# at once overwhelms Ray startup -> "raylet failed to startup / GCS overloaded"
# (this is NOT a system-RAM problem) and also exhausts a single GPU's VRAM.
# So we cap concurrency (MAX_PARALLEL) and space out launches (STAGGER_SEC).
#
# Tune MAX_PARALLEL by GPU VRAM, not system RAM:
#   1) run once with MAX_PARALLEL=1 and read system/peak_vram_gb in W&B,
#   2) set MAX_PARALLEL ~= floor(GPU_VRAM_GB / peak_vram_per_run), with headroom.
# IPPO/MAPPO (they train a critic) use more VRAM than GRPO, so start low.
# Override without editing this file, e.g.:  MAX_PARALLEL=2 bash run_experiments.sh
# ============================================================
MAX_PARALLEL="${MAX_PARALLEL:-6}"
STAGGER_SEC="${STAGGER_SEC:-10}"
ENVS="MultiGrid-Cluttered-Fixed-15x15 MultiGrid-FourRooms-15x15-v0"
SEEDS="42 1 7"
WB="multigrid-ippo"

launch() {
    # Block until fewer than MAX_PARALLEL jobs are running, then start in background.
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do sleep 5; done
    "$@" &
    sleep "$STAGGER_SEC"   # let this Ray cluster come up before starting the next
}

for env in $ENVS; do
    for s in $SEEDS; do
        launch python main.py --algorithm IPPO  --mode ppo  --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/IPPO/seed_$s"
    done
    for s in $SEEDS; do
        launch python main.py --algorithm MAPPO --mode ppo  --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/MAPPO/seed_$s"
    done
    for s in $SEEDS; do
        launch python main.py --algorithm IPPO  --mode grpo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/grpo_return/seed_$s" --use_return
    done
    for s in $SEEDS; do
        launch python main.py --algorithm IPPO  --mode grpo --seed $s --env_name "$env" --wandb_project "$WB" --save_path "/checkpoints/$env/grpo_global_norm_return/seed_$s" --global_norm --use_return
    done
done
wait
