# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RL config for Manipulation envs."""

from typing import Optional
from ml_collections import config_dict
from mujoco_playground._src import parahand


def brax_ppo_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = parahand.get_default_config(env_name)

  rl_config = config_dict.create(
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=env_config.action_repeat,
      reward_scaling=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(32, 32, 32, 32),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
      num_resets_per_eval=10,
  )
  if env_name == "ParaNontendonFR3Grasp":
    rl_config.normalize_observations=False
    rl_config.num_timesteps = 2_000_000_000
    rl_config.num_evals = 100
    rl_config.num_minibatches = 4
    rl_config.unroll_length = 32
    rl_config.num_updates_per_batch = 5
    rl_config.discounting = 0.99
    rl_config.gae_lambda = 0.95
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 0.005
    rl_config.desired_kl = 0.01
    rl_config.max_grad_norm = 1.0
    rl_config.num_envs = 4096
    rl_config.batch_size = 1024
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="state",
    )
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config
