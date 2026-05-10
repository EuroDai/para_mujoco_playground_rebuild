import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

# 引入基础环境类
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.parahand.para_nontendon_fr3 import para_nontendon_fr3_constants as consts

def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,        # 策略控制频率 50Hz
        sim_dt=0.002,        # 底层物理仿真频率 500Hz
        episode_length=256,  # 每个回合最大步数
        action_repeat=1,
        action_scale=0.05,    # 增量动作的缩放比例
        impl='warp', # 默认用warp，
        naconmax=30*8192, 
        naccdmax=30*8192, 
        njmax=160
    )
    return config

class ParaNontendonFR3Grasp(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
        xml_path=consts.GRASP_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
        
    def _post_init(self) -> None:
        home_key = self._mj_model.keyframe("home")
        self._init_q = jp.array(home_key.qpos, dtype=float)
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._robot_qids = mjx_env.get_qpos_ids(self.mj_model, consts.ALL_JOINTS)
        self._default_pose = self._init_q[self._robot_qids]


    def reset(self, rng: jax.Array) -> State:

        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pass

    def _get_reward(self, data: mjx.Data, info: dict) -> jax.Array:
        pass

    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        state = jp.concatenate([
        ])
        
        # 【Actor和Critic对等】：直接复用同一个 state
        return {
            "state": state,
            "privileged_state": state,
        }