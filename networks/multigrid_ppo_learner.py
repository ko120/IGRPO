"""Custom PPO Learner with separate actor and critic optimizers."""

from collections import defaultdict

import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_KL_KEY
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
import pdb
class GRPOAdvantageEstimation(GeneralAdvantageEstimation):

    def __init__(self, input_observation_space=None, input_action_space=None,
                 *, gamma, lambda_, global_norm=False, use_return=True):
        super().__init__(input_observation_space, input_action_space,
                         gamma=gamma, lambda_=lambda_)
        self.global_norm = global_norm
        self.use_return = use_return

    @override(ConnectorV2)
    def __call__(self, *, rl_module, episodes, batch, **kwargs):
        sa_episodes_list = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )

        # Group single-agent episodes by parent env episode.
        # Agents in the same MultiAgentEpisode share multi_agent_episode_id.
        groups = defaultdict(list)
        for ep in sa_episodes_list:
            groups[ep.multi_agent_episode_id].append(ep)

        ep_advantages = {}  # id(ep) -> np.ndarray of shape (len(ep),)

        for group_eps in groups.values():
            if self.global_norm:
                ep_returns = {}
                for ep in group_eps:
                    rewards = np.array(ep.get_rewards(), dtype=np.float32)
                    if self.use_return:
                        R = np.zeros(len(rewards), dtype=np.float32)
                        running = 0.0
                        for t in range(len(rewards) - 1, -1, -1):
                            running = rewards[t] + self.gamma * running
                            R[t] = running
                        ep_returns[id(ep)] = R
                    else:
                        ep_returns[id(ep)] = rewards

                all_returns = np.concatenate(list(ep_returns.values()), dtype=np.float32)
                mean = all_returns.mean()
                std = max(1e-4, float(all_returns.std()))

                for ep in group_eps:
                    ep_advantages[id(ep)] = (ep_returns[id(ep)] - mean) / std

            else:
                if self.use_return:
                    ep_returns_list = []
                    for ep in group_eps:
                        rewards = np.array(ep.get_rewards(), dtype=np.float32)
                        R = np.zeros(len(rewards), dtype=np.float32)
                        running = 0.0
                        for t in range(len(rewards) - 1, -1, -1):
                            running = rewards[t] + self.gamma * running
                            R[t] = running
                        ep_returns_list.append(R)
                    returns_matrix = np.array(ep_returns_list, dtype=np.float32)  # shape: (n_agents, T)
                    mean = returns_matrix.mean(axis=0)                    # shape: (T,)
                    std = np.maximum(1e-4, returns_matrix.std(axis=0))    # shape: (T,)
                    norm_matrix = (returns_matrix - mean) / std           # shape: (n_agents, T)
                    for ep, norm_r in zip(group_eps, norm_matrix):
                        ep_advantages[id(ep)] = norm_r
                else:
                    rewards_matrix = np.array(
                        [np.array(ep.get_rewards()) for ep in group_eps], dtype=np.float32
                    )  # shape: (n_agents, T)
                    mean = rewards_matrix.mean(axis=0)                    # shape: (T,)
                    std = np.maximum(1e-4, rewards_matrix.std(axis=0))    # shape: (T,)
                    norm_matrix = (rewards_matrix - mean) / std           # shape: (n_agents, T)
                    for ep, norm_r in zip(group_eps, norm_matrix):
                        ep_advantages[id(ep)] = norm_r

        # Write per-module advantages into the batch (same ordering as sa_episodes_list).
        device = None
        for mid in batch:
            module_eps = [e for e in sa_episodes_list if e.module_id in [None, mid]]
            if not module_eps:
                continue
            advantages = np.concatenate([ep_advantages[id(e)] for e in module_eps])
            batch[mid][Postprocessing.ADVANTAGES] = advantages
            batch[mid][Postprocessing.VALUE_TARGETS] = np.zeros_like(advantages)

            # Grab device from any existing tensor in this module's batch.
            if device is None:
                for v in batch[mid].values():
                    if hasattr(v, "device"):
                        device = v.device
                        break

        if self._numpy_to_tensor_connector is None:
            self._numpy_to_tensor_connector = NumpyToTensor(
                as_learner_connector=True, device=device
            )
        tensor_results = self._numpy_to_tensor_connector(
            rl_module=rl_module,
            batch={
                mid: {
                    Postprocessing.ADVANTAGES: batch[mid][Postprocessing.ADVANTAGES],
                    Postprocessing.VALUE_TARGETS: batch[mid][Postprocessing.VALUE_TARGETS],
                }
                for mid in batch
                if Postprocessing.ADVANTAGES in batch[mid]
            },
            episodes=episodes,
        )
        for mid, module_batch in tensor_results.items():
            batch[mid].update(module_batch)

        return batch


class BackwardEpisode(ConnectorV2):
    """Reverses batch along time axis after GAE so end-of-episode rewards appear first."""

    def __call__(self, *, rl_module, batch, episodes, explore=None, shared_data=None, **kwargs):
        for module_batch in batch.values():

            for key, val in module_batch.items():
                if isinstance(val, torch.Tensor):
                    module_batch[key] = val.flip(0)
  
        return batch


class MultiGridGRPOLearner(PPOTorchLearner):
    """GRPO Learner: group-relative policy optimization, no value function."""

    @override(PPOTorchLearner)
    def build(self):
        super().build()  # PPOLearner.build() appends GAE last

        if (
            self._learner_connector is not None
            and self.config.add_default_connectors_to_learner_pipeline
        ):
            # Swap out GAE for GRPO advantage estimation.
            connectors = self._learner_connector.connectors
            for i, connector in enumerate(connectors):
                if isinstance(connector, GeneralAdvantageEstimation):
                    connectors[i] = GRPOAdvantageEstimation(
                        gamma=self.config.gamma,
                        lambda_=self.config.lambda_,
                        global_norm=getattr(self.__class__, '_global_norm', False),
                        use_return=getattr(self.__class__, '_use_return', True),
                    )
                    break

            module_ids = list(self._module.keys())
            if module_ids and self._module[module_ids[0]].model_config.get("backward", False):
                self._learner_connector.append(BackwardEpisode())

    @override(PPOTorchLearner)
    def configure_optimizers_for_module(self, module_id, **kwargs):
        module = self._module[module_id]
        lr = self.config.lr

        # GRPO has no critic — only optimize actor parameters.
        if module.model_config.get("share_backbone", False):
            actor_params = list(module.parameters())
        else:
            actor_params = (
                list(module.image_layers.parameters())
                + list(module.pi_direction_layers.parameters())
                + list(module.pi_trunk.parameters())
                + list(module.pi_head.parameters())
            )

        self.register_optimizer(
            module_id=module_id,
            optimizer_name="actor",
            optimizer=torch.optim.Adam(actor_params, lr=lr, eps=1e-5),
            params=actor_params,
        )

    @override(PPOTorchLearner)
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()

        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid
        else:
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )
        if config.use_kl_loss: # default = True
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        adv = batch[Postprocessing.ADVANTAGES]

        surrogate_loss = torch.min(
            adv * logp_ratio,
            adv * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        # GRPO: no value function loss, no entropy loss.
        # entropy_coeff = self.entropy_coeff_schedulers_per_module[
        #     module_id
        # ].get_current_value()
        total_loss = possibly_masked_mean(-surrogate_loss)

        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,
        )
        return total_loss


class MultiGridPPOLearner(PPOTorchLearner):
    """PPO Learner with separate actor/critic optimizers."""

    @override(PPOTorchLearner)
    def build(self):
        super().build()  # appends GAE last via PPOLearner.build()
        if (
            self._learner_connector is not None
            and self.config.add_default_connectors_to_learner_pipeline
        ):
            module_ids = list(self._module.keys())
            backward = (
                self._module[module_ids[0]].model_config.get("backward", False)
                if module_ids else False
            )
            if backward:
                self._learner_connector.append(BackwardEpisode())


    @override(PPOTorchLearner)
    def configure_optimizers_for_module(self, module_id, **kwargs):
        module = self._module[module_id]
        lr = self.config.lr

        if module.model_config.get("share_backbone", False):
            # Single optimizer for all params: backbone gradients flow from
            # both policy and value losses and are updated together.
            all_params = list(module.parameters())
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="shared",
                optimizer=torch.optim.Adam(all_params, lr=lr, eps=1e-5),
                params=all_params,
            )
        else:
            actor_params = (
                list(module.image_layers.parameters())
                + list(module.pi_direction_layers.parameters())
                + list(module.pi_trunk.parameters())
                + list(module.pi_head.parameters())
            )
            critic_params = (
                list(module.vf_image_layers.parameters())
                + list(module.vf_direction_layers.parameters())
                + list(module.vf_trunk.parameters())
                + list(module.vf_head.parameters())
            )
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="actor",
                optimizer=torch.optim.Adam(actor_params, lr=lr, eps=1e-5),
                params=actor_params,
            )
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="critic",
                optimizer=torch.optim.Adam(critic_params, lr=lr, eps=1e-5),
                params=critic_params,
            )
