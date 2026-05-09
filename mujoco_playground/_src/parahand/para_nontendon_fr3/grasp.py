"""Grasp an object with a nontendon-parahand with fairino3 arm."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np

def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
    )
    return config

class ParaNontendonFR3Grasp(mjx_env.MjxEnv):
    def __init__(self):
        pass

    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        pass

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        state = jp.concatenate([

        ])

        privileged_state = jp.concatenate([
            state,
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _reward_fingers_to_object(self, data: mjx.Data) -> jax.Array:
        pass

    def _reward_good_finger_contact(self, data: mjx.Data) -> jax.Array:
        pass
