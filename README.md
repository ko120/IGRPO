
## Files

**`multiagent_metacontroller.py`** — the main agent class. wraps RLlib's PPO algo, handles the training loop, checkpointing, and wandb logging.

**`multigrid_rllib_env.py`** — RLlib env wrapper.


**`networks/multigrid_ppo_rl_module.py`** — actor/critic network.

**`networks/multigrid_ppo_learner.py`** — custom RLlib learner. 


## Algorithms

**IPPO** — each agent's critic sees only its own local obs.

**MAPPO** — critic sees all agents' views concatenated.

**GRPO** — no critic. advantages computed by normalizing returns across agents in the same episode. 
