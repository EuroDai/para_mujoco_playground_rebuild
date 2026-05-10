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
from mujoco_playground._src.parahand.para_nontendon_fr3 import base as para_nontendon_fr3_base
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

class ParaNontendonFR3Grasp(para_nontendon_fr3_base.ParaNontendonFR3Env):
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
        self._init_ctrl = jp.array(home_key.ctrl, dtype=float)
        self._init_qvel = jp.array(home_key.qvel, dtype=float)
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
        self._robot_qids = mjx_env.get_qpos_ids(self.mj_model, consts.ALL_JOINTS)
        self._default_pose = self._init_q[self._robot_qids]


    def reset(self, rng: jax.Array) -> State:
        data = mjx_env.make_data(
            self._mj_model,
            qpos=self._init_q,
            ctrl=self._init_ctrl,
            qvel=self._init_qvel,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        obs = jp.zeros(1)
        reward, done = jp.zeros(2)
        metrics = {}
        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(self._mjx_model.nu)
        }

        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )
        obs = self._get_obs(data, state.info)
        reward = jp.array(0.0)
        done = self._get_termination(data, state.info)
        metrics = state.metrics.copy()
        info = state.info.copy()
        info["step"] = info["step"] + 1
        info["last_act"] = action
        # ==============================================

        # 推荐使用 state.replace() 来生成新的 State，它是 Flax/Chex 结构体的标准用法
        # 这能确保其他没有改动的隐式属性也被安全地保留下来
        state = state.replace(
            data=data, 
            obs=obs, 
            reward=reward, 
            done=done, 
            metrics=metrics, 
            info=info
        )
        return state

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        return jp.zeros(1)

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict, metrics: dict) -> dict:
        return {}

    def _get_termination(self, data: mjx.Data, info: dict) -> jax.Array:
        return jp.array(0.0)