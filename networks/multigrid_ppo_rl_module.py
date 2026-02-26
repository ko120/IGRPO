from typing import Any, Dict, Optional

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

NUM_DIRECTIONS = 4
_IMG_CH = 3


class MultiGridPPORLModule(TorchRLModule, ValueFunctionAPI):
    """Actor/critic networks with optional shared CNN backbone."""

    @override(TorchRLModule)
    def setup(self):
        kernel_size = self.model_config.get("kernel_size", 3)
        fc_direction = self.model_config.get("fc_direction", 8)
        n_agents = self.model_config.get("n_agents", 3)
        algorithm = self.model_config.get("algorithm", "MAPPO")
        self._share_backbone = self.model_config.get("share_backbone", False)

        self._use_global_vf = (algorithm == "MAPPO")

        self._global_img_end  = _IMG_CH + n_agents * _IMG_CH
        self._local_dir_start = self._global_img_end
        self._local_dir_end   = self._local_dir_start + NUM_DIRECTIONS
        self._global_dir_end  = self._local_dir_end + n_agents * NUM_DIRECTIONS

        # Two conv layers with kernel_size (no padding): H_out = H - 2*(kernel_size-1)
        obs_h = self.observation_space.shape[0]
        h_out = obs_h - 2 * (kernel_size - 1)
        cnn_flat_size = 64 * h_out * h_out

        self.image_layers = nn.Sequential(
            nn.Conv2d(_IMG_CH, 32, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(cnn_flat_size, 64),
            nn.LeakyReLU(),
        )
        self.pi_direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, fc_direction),
            nn.ReLU(),
        )

        if not self._share_backbone:
            vf_img_ch = n_agents * _IMG_CH if self._use_global_vf else _IMG_CH
            vf_dir_ch = n_agents * NUM_DIRECTIONS if self._use_global_vf else NUM_DIRECTIONS
            self.vf_image_layers = nn.Sequential(
                nn.Conv2d(vf_img_ch, 32, (kernel_size, kernel_size)),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, (kernel_size, kernel_size)),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(cnn_flat_size, 64),
                nn.LeakyReLU(),
            )
            self.vf_direction_layers = nn.Sequential(
                nn.Linear(vf_dir_ch, fc_direction),
                nn.ReLU(),
            )

        self.pi_trunk = nn.Sequential(
            nn.Linear(64 + fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(64, self.action_space.n)
        self.vf_head = nn.Linear(64, 1)

        if not self._share_backbone:
            self.vf_trunk = nn.Sequential(
                nn.Linear(64 + fc_direction, 192),
                nn.ReLU(),
                nn.Linear(192, 64),
                nn.ReLU(),
            )

    def _backbone_features(self, batch):
        obs = batch[Columns.OBS]
        image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
        img = self.image_layers(image)
        local_dir = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        dir_feat = self.pi_direction_layers(local_dir)
        return torch.cat([img, dir_feat], dim=-1)

    def _pi_features(self, batch):
        if self._share_backbone:
            return self.pi_trunk(self._backbone_features(batch))
        obs = batch[Columns.OBS]
        image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
        img = self.image_layers(image)
        local_dir = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        dir_feat = self.pi_direction_layers(local_dir)
        return self.pi_trunk(torch.cat([img, dir_feat], dim=-1))

    def _vf_features(self, batch):
        if self._share_backbone:
            return self.pi_trunk(self._backbone_features(batch))
        obs = batch[Columns.OBS]
        if self._use_global_vf:
            image = obs[:, :, :, _IMG_CH:self._global_img_end].permute(0, 3, 1, 2).float()
            dir_input = obs[:, 0, 0, self._local_dir_end:self._global_dir_end].float()
        else:
            image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
            dir_input = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        img = self.vf_image_layers(image)
        dir_feat = self.vf_direction_layers(dir_input)
        return self.vf_trunk(torch.cat([img, dir_feat], dim=-1))

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch))}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # When share_backbone=True the CNN runs once and both heads reuse it.
        if self._share_backbone:
            trunk = self.pi_trunk(self._backbone_features(batch))
            return {
                Columns.ACTION_DIST_INPUTS: self.pi_head(trunk),
                Columns.EMBEDDINGS: trunk,
            }
        return {
            Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch)),
            Columns.EMBEDDINGS: self._vf_features(batch),
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is None:
            embeddings = self._vf_features(batch)
        return self.vf_head(embeddings).squeeze(-1)
    


class MultiGridGRPORLModule(TorchRLModule, ValueFunctionAPI):
    """Actor/critic networks with optional shared CNN backbone."""

    @override(TorchRLModule)
    def setup(self):
        kernel_size = self.model_config.get("kernel_size", 3)
        fc_direction = self.model_config.get("fc_direction", 8)
        n_agents = self.model_config.get("n_agents", 3)
        algorithm = self.model_config.get("algorithm", "MAPPO")
        self._share_backbone = self.model_config.get("share_backbone", False)

        self._global_img_end  = _IMG_CH + n_agents * _IMG_CH
        self._local_dir_start = self._global_img_end
        self._local_dir_end   = self._local_dir_start + NUM_DIRECTIONS
        self._global_dir_end  = self._local_dir_end + n_agents * NUM_DIRECTIONS

        # Two conv layers with kernel_size (no padding): H_out = H - 2*(kernel_size-1)
        obs_h = self.observation_space.shape[0]
        h_out = obs_h - 2 * (kernel_size - 1)
        cnn_flat_size = 64 * h_out * h_out

        self.image_layers = nn.Sequential(
            nn.Conv2d(_IMG_CH, 32, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(cnn_flat_size, 64),
            nn.LeakyReLU(),
        )
        self.pi_direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, fc_direction),
            nn.ReLU(),
        )



        self.pi_trunk = nn.Sequential(
            nn.Linear(64 + fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(64, self.action_space.n)

    def _pi_features(self, batch):
        obs = batch[Columns.OBS]
        image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
        img = self.image_layers(image)
        local_dir = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        dir_feat = self.pi_direction_layers(local_dir)
        return self.pi_trunk(torch.cat([img, dir_feat], dim=-1))


    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch))}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch))}

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # GRPO does not use a value function — return zeros so the PPO pipeline
        # does not crash if it calls this (e.g. for logging).
        B = batch[Columns.OBS].shape[0]
        return torch.zeros(B, device=batch[Columns.OBS].device)
