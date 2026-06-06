# IGRPO — Single-Trajectory GRPO for Cooperative MARL

Research code for the paper **"Single Trajectory GRPO for Cooperative Multi-Agent
Reinforcement Learning"** (Brian Ko, UW — see `latex/main.tex`, PDF in repo root).

**Core idea:** Standard GRPO forms its advantage baseline from *many trajectories*
sampled from the same prompt. In homogeneous cooperative MARL you don't have that —
so this work treats **the N agents within a single episode as the GRPO "group"** and
normalizes their returns against each other. No critic, no GAE, one trajectory.

**Claim:** Matches IPPO performance with up to **8× less VRAM** (critic-free).

---

## Quickstart

```bash
# Train one config (see run_experiments.sh for the full sweep)
python main.py --algorithm IPPO --mode grpo --seed 42 \
    --env_name MultiGrid-Cluttered-Fixed-15x15 \
    --wandb_project multigrid-ippo \
    --save_path /checkpoints/cluttered_new/grpo_global_norm_return \
    --global_norm --use_return

python main.py ... --debug          # disables wandb (use for local smoke tests)
python main.py ... --visualize --load_checkpoint_from <ckpt>   # renders a video
```

- `run_experiments.sh` — the paper's GRPO sweep (4 runs: {Cluttered, Meetup} ×
  {per-timestep, global-norm}, all `--use_return`).
- `run_visualize.sh` — renders trajectories from a checkpoint.
- Deps already installed in this env: `ray 2.55`, `gymnasium 1.2`, torch, wandb.

### ⚠️ Read before running
1. **`num_gpus_per_learner=1` is hardcoded** at `multiagent_metacontroller.py:95`.
   This machine is macOS/CPU (no CUDA) → training **will fail** until you set it to
   `0`. (`main.py` device detection is unrelated; this is the RLlib learner config.)
2. **`requirements.txt` is incomplete** — it omits `ray[rllib]` and `gymnasium`,
   which the whole codebase imports. `setup.sh` will NOT reproduce a working env.
   They happen to be installed here already.
3. **wandb is online by default** (`utils.py:56`). Use `--debug` to avoid logging.

---

## How it works (data flow)

```
main.py → MultiAgent (multiagent_metacontroller.py) → RLlib PPOConfig.build_algo()
        → algo.train() loop   [RLlib does all sampling + optimization]
```

Training uses **RLlib's new API stack** (RLModule + Learner + ConnectorV2). The
`train()` loop just calls `algo.train()` repeatedly and logs. `get_actions()` /
`run_one_episode()` in the metacontroller are **only for visualization**, not training.

### The novelty lives in ONE place
`networks/multigrid_ppo_learner.py` → **`GRPOAdvantageEstimation`** (a ConnectorV2).
`MultiGridGRPOLearner.build()` swaps RLlib's default GAE connector for this one. It:
1. Groups single-agent episodes by their shared `multi_agent_episode_id` (the group).
2. Computes per-agent returns, normalizes them across the group → advantage.
3. Writes zeros to `VALUE_TARGETS` (no critic). Loss = clipped surrogate + KL only
   (no value loss, no entropy bonus) in `compute_loss_for_module`.

**Four GRPO advantage variants** (CLI flags `--global_norm`, `--use_return`):

| `--use_return` | `--global_norm` | Baseline (μ, σ) computed over… | Signal |
|---|---|---|---|
| ✓ (default) | ✗ (default) | agents, **per timestep** | discounted return-to-go |
| ✓ | ✓ | agents **and** all timesteps (scalar) | discounted return-to-go |
| ✗ | ✗ | agents, per timestep | raw per-step reward |
| ✗ | ✓ | agents and all timesteps | raw per-step reward |

> Defaults are effectively `use_return=True`, `global_norm=False` (set via `getattr`
> fallbacks in `multiagent_metacontroller.py:68-69`; argparse defaults are `None` so
> config wins). The `--use_return` help text in `main.py` mislabels the default — ignore it.

### Observation trick (enables IPPO/MAPPO from one obs space)
`multigrid_rllib_env.py` packs **one Box per agent**:
`[local_img(3) | global_imgs(N·3) | local_dir(4) | global_dir(N·4)]`
(N=3 → 28 channels). The RLModule **slices channels** to feed each head:
- **Actor** always sees local only.
- **Critic** sees local (IPPO) or global (MAPPO), switched by
  `_use_global_vf = (algorithm=="MAPPO")` in `multigrid_ppo_rl_module.py:27`.
- GRPO has no critic, so IPPO/MAPPO is moot for `--mode grpo` (actor-only).

If you change `n_agents`, view size, or this layout, **keep the env wrapper and the
RLModule channel slicing in sync** or it breaks silently.

---

## File map

| Path | Role |
|---|---|
| `main.py` | CLI args, config init, dispatch train/visualize. |
| `multiagent_metacontroller.py` | `MultiAgent`: builds `PPOConfig`, train loop, ckpt, wandb, viz. Picks learner by mode. |
| `networks/multigrid_ppo_learner.py` | **Core.** `GRPOAdvantageEstimation`, `MultiGridGRPOLearner`, `MultiGridPPOLearner`, `BackwardEpisode`. |
| `networks/multigrid_ppo_rl_module.py` | `MultiGridPPORLModule` (actor/critic, channel slicing, IPPO/MAPPO). |
| `multigrid_rllib_env.py` | `MultiAgentEnv` wrapper: obs packing/splitting, step/reset. |
| `utils.py` | Config merge (default←domain←mode), `make_env`, viz plotting, video. |
| `config/` | `default.yaml` + `mode/{ppo,grpo}.yaml` + `domain/*.yaml`. |
| `envs/gym_multigrid/` | The gridworld. `multigrid.py` (base, 7 actions, shuffled agent order) + `multigrid_envs/`. |
| `latex/` | The paper (compile-ready). |

**Dead / unused code (don't be fooled):**
- `networks/multigrid_ppo_rl_module.py:149` `MultiGridGRPORLModule` — **never used**;
  GRPO mode still uses `MultiGridPPORLModule` and only swaps the *Learner*.
- `networks/multigrid_network.py` — legacy DQN-style net, not in the RLlib path.

---

## Environments (both N=3 agents)

| Env | Obs | Reward | Notes |
|---|---|---|---|
| `MultiGrid-Cluttered-Fixed-15x15` | partial **5×5×3** | **sparse**: only on reaching goal, scaled by lateness (minigrid `_reward()`) | fixed walls/goal, randomized agent starts, `max_steps=100`, 30 clutter. |
| `MultiGrid-Meetup-Empty-15x15-v0` | full **15×15×3** (`fully_observed`) | **dense**: per-step Δdistance to best meetup point, +1 when all meet | `max_steps=250`, 3 goal doors. |

(Many more variants are registered in `envs/gym_multigrid/multigrid_envs/*.py` but
only these two are used in experiments.)

---

## Config system
`utils.generate_parameters()` merges, lowest→highest priority:
`config/default.yaml` → `config/domain/<env>.yaml` → `config/mode/<ppo|grpo>.yaml`,
then CLI flags override in `main.py`. Key knobs: `lr 1e-3`, `gamma 0.99`,
`train_batch_size 4000`, `minibatch_size 250`, `num_epochs 4`, `clip 0.2`,
`n_episodes 30000`. `share_backbone`: **True for ppo, False for grpo**.
`num_env_runners: 0` (sampling on the local worker).

---

## Gotchas cheat-sheet
- GPU hardcoded (see above) — first thing to fix for CPU runs.
- `requirements.txt` missing `ray[rllib]`, `gymnasium` (see above).
- `--save_path /foo` → leading `/` stripped, joined to repo root → `<repo>/foo`
  (`main.py:92`). So `/checkpoints/...` writes inside the repo.
- `--backward` / `BackwardEpisode`: experimental; reverses batch on the time axis
  after advantage est. Requires `shuffle_batch_per_epoch=False` (set when backward).
- Single seed (42) everywhere; paper explicitly lists single-run + single-env as
  limitations → multi-seed / more envs is the obvious next experiment.

## Likely next tasks
- Make `num_gpus_per_learner` config-driven (CPU support).
- Add `ray[rllib]`/`gymnasium` to `requirements.txt`.
- Multi-seed runs + aggregation; add a 3rd env.
- Wire up `MultiGridGRPORLModule` (true critic-free module) or delete it.
