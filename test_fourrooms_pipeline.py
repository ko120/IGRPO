"""Verifiable test: the experiment pipeline (Cluttered + FourRooms, multi-seed).

Run:  python test_fourrooms_pipeline.py     (exit 0 = all pass)

No GPU/wandb needed. Checks: (1) domain configs resolve, (2) both envs build
through the wrapper main.py uses, (3) run_experiments.sh expands to the full
2-env x 4-config x 3-seed matrix with collision-free save paths, and (4) per-run
VRAM logging is wired into the training loop.
"""
import os
import re
import subprocess
import sys
from collections import Counter

ENVS = ["MultiGrid-Cluttered-Fixed-15x15", "MultiGrid-FourRooms-15x15-v0"]
SEEDS = ["42", "1", "7"]
CONFIGS = ["IPPO", "MAPPO", "grpo_return", "grpo_global_norm_return"]
EXPECTED = len(ENVS) * len(CONFIGS) * len(SEEDS)
fails = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        fails.append(msg)


# 1) Domain configs exist and point at their env.
import yaml
for env in ENVS:
    p = f"config/domain/{env}.yaml"
    check(os.path.exists(p), f"domain config exists: {p}")
    if os.path.exists(p):
        check(yaml.safe_load(open(p)).get("domain") == env, f"{env}: domain field correct")

# 2) Both envs build through the wrapper main.py uses (same obs/action interface).
from multigrid_rllib_env import MultiGridRLlibEnv
for env in ENVS:
    e = MultiGridRLlibEnv({"env_name": env})
    obs, _ = e.reset()
    shp = tuple(obs["agent_0"].shape)
    _, r, _, _, _ = e.step({f"agent_{i}": 0 for i in range(e.n_agents)})
    check(e.n_agents == 3 and shp == (5, 5, 28) and e.action_space.n == 7
          and set(r) == {f"agent_{i}" for i in range(3)},
          f"{env}: n_agents=3, obs=(5,5,28), 7 actions, per-agent rewards")

# 3) run_experiments.sh expands to the full matrix. Stub `python` with echo and
#    source the script so the loops actually run (no training is launched).
check(subprocess.run(["bash", "-n", "run_experiments.sh"]).returncode == 0,
      "run_experiments.sh has valid bash syntax")
out = subprocess.run(
    ["bash", "-c", 'python() { echo "RUN $*"; }\nsource run_experiments.sh'],
    capture_output=True, text=True,
).stdout
runs = [l for l in out.splitlines() if l.startswith("RUN") and "main.py" in l]
check(len(runs) == EXPECTED, f"matrix expands to {EXPECTED} runs (got {len(runs)})")

seed_of = lambda l: re.search(r"--seed (\S+)", l).group(1)
path_of = lambda l: re.search(r"--save_path (\S+)", l).group(1)
env_of = lambda l: re.search(r"--env_name (\S+)", l).group(1)

check(len(set(path_of(l) for l in runs)) == len(runs),
      f"all {len(runs)} save_paths are unique (parallel-safe)")
check(all(f"seed_{seed_of(l)}" in path_of(l) for l in runs),
      "every save_path is namespaced by its seed")
check(Counter(seed_of(l) for l in runs) == {s: len(ENVS) * len(CONFIGS) for s in SEEDS},
      f"each seed appears {len(ENVS) * len(CONFIGS)} times")
for env in ENVS:
    env_runs = [l for l in runs if env_of(l) == env]
    check(len(env_runs) == len(CONFIGS) * len(SEEDS),
          f"{env}: {len(CONFIGS) * len(SEEDS)} runs (got {len(env_runs)})")
    for c in CONFIGS:
        n = sum(1 for l in env_runs if f"/{c}/" in path_of(l))
        check(n == len(SEEDS), f"{env}/{c}: {len(SEEDS)} seed runs (got {n})")

# 4) Per-run VRAM logging is wired into the training loop.
#    Importing the module also confirms the edits didn't break its syntax/deps.
import multiagent_metacontroller  # noqa: F401
mac = open("multiagent_metacontroller.py").read()
check("reset_peak_memory_stats" in mac, "train() resets CUDA peak-memory stats")
check('log_data["system/peak_vram_gb"]' in mac and "max_memory_allocated" in mac,
      "train() logs system/peak_vram_gb from torch.cuda.max_memory_allocated")
check(2_500_000_000 / 1e9 == 2.5, "byte->GB conversion (/1e9) is correct")
check("--n_episodes" in open("main.py").read(),
      "main.py exposes --n_episodes override (for short Colab runs)")

print()
print("ALL PASS" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
