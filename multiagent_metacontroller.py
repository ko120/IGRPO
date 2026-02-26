import numpy as np
import os
import torch
import wandb

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from networks.multigrid_ppo_rl_module import MultiGridPPORLModule
from networks.multigrid_ppo_learner import MultiGridPPOLearner

from multigrid_rllib_env import MultiGridRLlibEnv
from utils import plot_single_frame, make_video


def _checkpoint_path(result):
    """Extract a plain directory path from an algo.save() result.

    RLlib may return a Checkpoint object or a _TrainingResult wrapping one.
    """
    # _TrainingResult wraps a Checkpoint
    checkpoint = getattr(result, 'checkpoint', result)
    # Checkpoint object has a .path attribute
    path = getattr(checkpoint, 'path', None)
    if path is not None:
        return str(path)
    return str(checkpoint)


class MultiAgent():
    """RLlib PPO agent with shared policy across all agents (CTDE)."""

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.debug = debug
        self.device = device
        self.n_agents = env.n_agents
        self.model_others = getattr(config, 'model_others', False)
        self.use_local_obs = getattr(config, 'algorithm', 'MAPPO') == 'IPPO'
        self.total_steps = 0
        self.total_episodes = 0

        # Store the raw env for visualization (RLlib manages its own env copies)
        self._raw_env = env

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4)

        env_name = getattr(config, 'domain', 'MultiGrid-Cluttered-Fixed-15x15')
        use_local_obs = self.use_local_obs
        register_env(
            "multigrid",
            lambda cfg: MultiGridRLlibEnv({"env_name": env_name}),
        )
        model_config = {
            "kernel_size": config.kernel_size,
            "fc_direction": config.fc_direction,
            "n_agents": env.n_agents,
            "algorithm": getattr(config, "algorithm", "IPPO"),
            "share_backbone": getattr(config, "share_backbone", False),
            "backward": getattr(config, "backward", False),
        }
        if config.mode == 'grpo':
            from networks.multigrid_ppo_learner import MultiGridGRPOLearner
            MultiGridGRPOLearner._global_norm = getattr(config, 'global_norm', False)
            MultiGridGRPOLearner._use_return = getattr(config, 'use_return', True)
            learner_class = MultiGridGRPOLearner
            rl_module_spec = RLModuleSpec(
                module_class=MultiGridPPORLModule,
                model_config=model_config,
            )
        else:
            learner_class = MultiGridPPOLearner
            rl_module_spec = RLModuleSpec(
                module_class=MultiGridPPORLModule,
                model_config=model_config,
            )

        num_env_runners = getattr(config, 'num_env_runners', 0)
        self.algo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment("multigrid")
            .env_runners(
                num_env_runners=num_env_runners,
            )
            .learners(
                learner_class=learner_class,
                num_gpus_per_learner=1,
            )
            .training(
                lr=getattr(config, 'lr', 0.0003),
                gamma=getattr(config, 'gamma', 0.99),
                lambda_=getattr(config, 'lambda_', 0.95),
                clip_param=getattr(config, 'clip_param', 0.2),
                vf_loss_coeff=getattr(config, 'vf_loss_coeff', 0.5),
                entropy_coeff=getattr(config, 'entropy_coeff', 0.01),
                train_batch_size_per_learner=getattr(config, 'train_batch_size', 4000),
                minibatch_size=getattr(config, 'minibatch_size', 250),
                num_epochs=getattr(config, 'num_epochs', 4),
                grad_clip=getattr(config, 'grad_clip', 0.5),
                shuffle_batch_per_epoch=False if getattr(config, 'backward', False) else True)
            .multi_agent(
                # CTDE: ALL agents map to the SAME shared policy
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
            .rl_module(
                # Use custom RLModule that follows multigrid_network.py architecture.
                # RLlib auto-wraps this in a MultiRLModule for multi-agent.
                rl_module_spec=rl_module_spec
            )
        )
        if training:
            self.algo = self.algo_config.build_algo()

    def get_agent_state(self, state, agent_id):
        return {
            'image': np.array(state['image'][agent_id], dtype=np.uint8),
            'direction': np.array(state['direction'], dtype=np.uint8),
        }

    def _encode_agent_obs(self, state, agent_id):
        from multigrid_rllib_env import NUM_DIRECTIONS
        all_images = state['image']
        directions = state['direction']
        image = np.array(all_images[agent_id], dtype=np.uint8)
        h, w = image.shape[0], image.shape[1]

        local_img = image.astype(np.float32)

        global_imgs = np.concatenate(
            [np.array(all_images[a], dtype=np.float32) for a in range(self.n_agents)],
            axis=-1,
        )

        local_dir = np.zeros(NUM_DIRECTIONS, dtype=np.float32)
        local_dir[int(directions[agent_id])] = 1.0

        global_dir = np.zeros(self.n_agents * NUM_DIRECTIONS, dtype=np.float32)
        for a, d in enumerate(directions):
            global_dir[a * NUM_DIRECTIONS + int(d)] = 1.0

        dir_onehot = np.concatenate([local_dir, global_dir])
        dir_channels = np.broadcast_to(
            dir_onehot[np.newaxis, np.newaxis, :], (h, w, len(dir_onehot))
        ).copy()
        return np.concatenate([local_img, global_imgs, dir_channels], axis=-1)

    def get_actions(self, state):
        module = self.algo.get_module("shared_policy")

        obs_list = [self._encode_agent_obs(state, i) for i in range(self.n_agents)]
        obs_batch = torch.FloatTensor(np.stack(obs_list, axis=0))

        with torch.no_grad():
            result = module.forward_inference({Columns.OBS: obs_batch})

        logits = result[Columns.ACTION_DIST_INPUTS]  # (n_agents, n_actions)
        actions = torch.distributions.Categorical(logits=logits).sample()
        return [int(a) for a in actions]

    def run_one_episode(self, env, episode, log=True, train=True,
                        save_model=True, visualize=False):
        state = env.reset()
        done = False
        t = 0
        rewards = []

        if visualize:
            viz_data = self.init_visualization_data(env, state)

        while not done:
            self.total_steps += 1
            t += 1

            actions = self.get_actions(state)
            next_state, reward_list, done, info = env.step(actions)
            rewards.append(reward_list)

            if visualize:
                viz_data = self.add_visualization_data(
                    viz_data, env, state, actions, next_state)

            state = next_state

        rewards = np.array(rewards)  # (T, n_agents)

        if log:
            self.log_one_episode(episode, t, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        if save_model:
            self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = rewards
            return viz_data

    def train(self, env):
        import time
        n_episodes = getattr(self.config, 'n_episodes', 100000)
        visualize_every = getattr(self.config, 'visualize_every', 10000)
        save_every = getattr(self.config, 'save_model_episode', 5000)
        log_every = getattr(self.config, 'log_episode', 100)
        print_every = getattr(self.config, 'print_every', 50)

        # Build a unique run-level save directory once so all checkpoints
        # from this run go into the same folder and never overwrite each other.
        _base_save_path = getattr(self.config, 'save_path', None)
        run_save_path = os.path.join(_base_save_path, f"run_{int(time.time())}") if _base_save_path else None
        # Estimate how many algo.train() iterations we need.
        # Each train() call collects train_batch_size steps.
        train_batch_size = getattr(self.config, 'train_batch_size', 4000)
        env_max_steps = getattr(self._raw_env, 'max_steps', 100)
        est_episodes_per_iter = max(1, train_batch_size // 100)

        iteration = 0
        while self.total_episodes < n_episodes:
            iteration += 1

            result = self.algo.train()

            episodes_this_iter = result.get(
                "num_episodes_lifetime", self.total_episodes + est_episodes_per_iter
            ) - self.total_episodes
            if episodes_this_iter <= 0:
                episodes_this_iter = est_episodes_per_iter
            self.total_episodes += episodes_this_iter
            self.total_steps = result.get("num_env_steps_sampled_lifetime", self.total_steps)

            env_runners = result.get("env_runners", {})
            mean_reward = env_runners.get("episode_return_mean", 0.0)
            episode_len = env_runners.get("episode_len_mean", 0.0)
            agent_returns = env_runners.get("agent_episode_returns_mean", {})

            if self.total_episodes % log_every < episodes_this_iter:
                log_data = {
                    "episode/x_axis": self.total_episodes,
                    "episode/collective_reward_mean": mean_reward,
                    "episode/episode_length_mean": episode_len,
                    "step/x_axis": self.total_steps,
                    "step/collective_reward_mean": mean_reward,
                }
                for agent_id, agent_reward in agent_returns.items():
                    log_data[f"episode/{agent_id}_reward_mean"] = agent_reward
                learners = result.get("learners", {})
                shared = learners.get("shared_policy", {})
                if shared:
                    log_data["train/policy_loss"] = shared.get("policy_loss", 0.0)
                    log_data["train/entropy"] = shared.get("entropy", 0.0)
                    log_data["train/kl_loss"] = shared.get("mean_kl_loss", 0.0)
                wandb.log(log_data)

            if iteration % max(1, print_every // est_episodes_per_iter) == 0:
                agent_str = " | ".join(
                    f"{aid}: {r:.3f}" for aid, r in sorted(agent_returns.items())
                )
                print(
                    f"Total steps: {self.total_steps} \t "
                    f"Episodes: ~{self.total_episodes} \t "
                    f"Mean reward: {mean_reward:.3f} \t "
                    f"Mean ep len: {episode_len:.1f} \t "
                    f"Per-agent: [{agent_str}]"
                )

            if self.total_episodes % save_every < episodes_this_iter:
                ep_save_path = os.path.join(run_save_path, f"ep_{self.total_episodes}") if run_save_path else None
                result = self.algo.save(ep_save_path) if ep_save_path else self.algo.save()
                checkpoint_dir = _checkpoint_path(result)
                print(f"Checkpoint saved at episode ~{self.total_episodes}: {checkpoint_dir}")

            # Visualization episode
            # if (self.total_episodes % visualize_every < episodes_this_iter
            #         and self.total_episodes > 0):
            #     print(f"Running visualization at episode ~{self.total_episodes}...")
            #     viz_data = self.run_one_episode(
            #         self._raw_env, self.total_episodes, log=False,
            #         train=False, save_model=False, visualize=True)
            #     self.visualize(
            #         self._raw_env,
            #         self.config.mode + '_training_step' + str(self.total_episodes),
            #         viz_data=viz_data)

        env.close()

        final_save_path = os.path.join(run_save_path, "final") if run_save_path else None
        final_checkpoint = self.algo.save(final_save_path) if final_save_path else self.algo.save()
        # RLlib may return a _TrainingResult object instead of a plain path string.
        checkpoint_path = _checkpoint_path(final_checkpoint)
        print(f"Final model saved: {checkpoint_path}")
        # artifact = wandb.Artifact(name="final_model", type="model")
        # artifact.add_dir(checkpoint_path)
        # wandb.log_artifact(artifact)
        # print("Final model uploaded to wandb.")

        self.algo.stop()

    def log_one_episode(self, episode, t, rewards):
        collective_reward = np.sum(rewards)
        per_agent_rewards = np.sum(rewards, axis=0)

        log_data = {
            "episode/x_axis": episode,
            "episode/collective_reward": collective_reward,
            "episode/episode_length": t,
        }
        for i in range(self.n_agents):
            log_data[f"episode/agent_{i}_reward"] = per_agent_rewards[i]

        wandb.log(log_data)

    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0 and episode > 0:
            save_path = getattr(self.config, 'save_path', None)
            result = self.algo.save(save_path) if save_path else self.algo.save()
            checkpoint_dir = _checkpoint_path(result)
            print(f"Checkpoint saved at episode {episode}: {checkpoint_dir}")

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None,
        }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image'])
             for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(
                self.get_action_predictions(next_state))
        return viz_data

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False,
                visualize=True)

        video_path = os.path.join(
            video_dir, self.config.experiment_name, self.config.model_name)

        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(
                t, viz_data, action_dict, video_path, self.config.model_name)
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path,
                            model_name):
        plot_single_frame(
            t,
            viz_data['full_images'][t],
            viz_data['agents_partial_images'][t],
            viz_data['actions'][t],
            viz_data['rewards'],
            action_dict,
            video_path,
            self.config.model_name,
            predicted_actions=viz_data['predicted_actions'],
            all_actions=viz_data['actions'])

    def load_models(self, model_path=None):
        """Load a saved RLlib checkpoint."""
        if model_path is not None:
            self.algo = self.algo_config.build_algo()
            self.algo.restore(model_path)
        else:
            print("No model path provided, using current algo state.")
