"""RLlib MultiAgentEnv wrapper for the multigrid environment."""

import gymnasium
import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import pdb 
NUM_DIRECTIONS = 4


class MultiGridRLlibEnv(MultiAgentEnv):

    def __init__(self, config=None):
        config = config or {}
        env_name = config.get("env_name", "MultiGrid-Cluttered-Fixed-15x15")
        self.use_local_obs = config.get("use_local_obs", False)

        from envs import gym_multigrid
        from envs.gym_multigrid import multigrid_envs 
        self._env = gym.make(env_name)

        self.n_agents = self._env.n_agents
        self._agent_ids = {f"agent_{i}" for i in range(self.n_agents)}
        self.agents = self.possible_agents = list(self._agent_ids)

        # Observation layout:
        #   local image    (3)           - actor
        #   global images  (n_agents*3)  - MAPPO critic
        #   local dir      (4)           - actor + IPPO critic
        #   global dir     (n_agents*4)  - MAPPO critic
        agent_view = self._env.agent_view_size
        obs_channels = (3 + self.n_agents * 3) + (NUM_DIRECTIONS + self.n_agents * NUM_DIRECTIONS)
        single_obs_space = gymnasium.spaces.Box(
            low=0.0, high=255.0,
            shape=(agent_view, agent_view, obs_channels),
            dtype=np.float32,
        )
        # multigrid uses Box action space for joint actions; get count from Actions enum
        n_actions = len(self._env.actions)
        single_action_space = gymnasium.spaces.Discrete(n_actions)

        self.observation_spaces = {
            f"agent_{i}": single_obs_space for i in range(self.n_agents)
        }
        self.action_spaces = {
            f"agent_{i}": single_action_space for i in range(self.n_agents)
        }
        self.observation_space = single_obs_space
        self.action_space = single_action_space

        super().__init__()

    def _encode_obs(self, image, all_images, directions, agent_view, agent_index):
        """Encode obs into a single (H, W, C) tensor.

        Layout: [local image (3)] + [global images (n_agents*3)]
              + [local dir (4)]   + [global dir (n_agents*4)]
        """
        h, w = agent_view, agent_view

        local_img = image.astype(np.float32)

        global_imgs = np.concatenate(
            [np.array(all_images[a], dtype=np.float32) for a in range(self.n_agents)],
            axis=-1,
        )

        local_dir = np.zeros(NUM_DIRECTIONS, dtype=np.float32)
        local_dir[int(directions[agent_index])] = 1.0

        global_dir = np.zeros(self.n_agents * NUM_DIRECTIONS, dtype=np.float32)
        for a, d in enumerate(directions):
            global_dir[a * NUM_DIRECTIONS + int(d)] = 1.0

        dir_onehot = np.concatenate([local_dir, global_dir])
        dir_channels = np.broadcast_to(
            dir_onehot[np.newaxis, np.newaxis, :], (h, w, len(dir_onehot))
        ).copy()
        return np.concatenate([local_img, global_imgs, dir_channels], axis=-1)

    def _split_obs(self, obs):
        """Convert multigrid stacked obs into per-agent encoded obs."""
        images = obs["image"]
        directions = obs["direction"]
        agent_view = self._env.agent_view_size

        agent_obs = {}
        for i in range(self.n_agents):
            agent_obs[f"agent_{i}"] = self._encode_obs(
                np.array(images[i], dtype=np.uint8), images, directions, agent_view, i
            )
        return agent_obs

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return self._split_obs(obs), {}

    def step(self, action_dict):
        actions = [
            int(action_dict.get(f"agent_{i}", 0))
            for i in range(self.n_agents)
        ]

        obs, rewards, done, info = self._env.step(actions)

        agent_obs = self._split_obs(obs)
        agent_rewards = {
            f"agent_{i}": float(rewards[i]) for i in range(self.n_agents)
        }
        agent_terminateds = {f"agent_{i}": done for i in range(self.n_agents)}
        agent_terminateds["__all__"] = done
        agent_truncateds = {f"agent_{i}": False for i in range(self.n_agents)}
        agent_truncateds["__all__"] = False
        agent_infos = {f"agent_{i}": {} for i in range(self.n_agents)}

        return agent_obs, agent_rewards, agent_terminateds, agent_truncateds, agent_infos

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def get_obs_render(self, obs_image):
        return self._env.get_obs_render(obs_image)

    def close(self):
        self._env.close()

    @property
    def unwrapped_env(self):
        return self._env
