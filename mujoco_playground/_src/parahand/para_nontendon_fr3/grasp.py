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
        action_scale=0.01,    # 增量动作的缩放比例
        impl='warp', # 默认用warp，
        naconmax=30 * 8192, 
        # naccdmax=240*8192, 
        njmax=500,
        reward_config=config_dict.create(
            scales=config_dict.create(
                fingertip_approach=1.0,
                good_finger_contact=0.5,
                position_tracking=2.0,
                success=10.0,
                action_l2=-0.005,
                action_rate_l2=-0.005,
                termination=-1.0,
            )
        )
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
        self._all_qids = mjx_env.get_qpos_ids(self.mj_model, consts.ALL_JOINTS)
        self._all_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.ALL_JOINTS)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._cube_body_id = self._mj_model.body("cube").id
        self._fingertip_site_ids = np.array([self._mj_model.site(n).id for n in consts.FINGERTIP_SITES])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._target_site_id = self._mj_model.site(consts.TARGET_SITE).id
        self._default_pose = self._init_q[self._all_qids]
        self._default_cube_pose = self._init_q[self._cube_qids]


    def reset(self, rng: jax.Array) -> State:
        rng, pos_rng, cube_rng = jax.random.split(rng, 3)

        q_noise = jax.random.normal(pos_rng, (self.mjx_model.nu,))
        q_robot = jp.clip(self._default_pose + 0.05 * q_noise, self._lowers, self._uppers)

        ctrl = jp.clip(self._init_ctrl + 0.05 * q_noise, self._lowers, self._uppers)
        
        q_cube = self._default_cube_pose.copy()
        q_cube = q_cube.at[:3].set(
            jax.random.uniform(
                cube_rng, (3,), minval=jp.array([-0.05, -0.05, 0.0]), maxval=jp.array([0.05, 0.05, 0.0])
            ) 
            + self._default_cube_pose[:3]
        )

        q_cube = q_cube.at[3:].set(para_nontendon_fr3_base.uniform_quat(cube_rng))
        ctrl = jp.clip(self._init_ctrl + 0.05 * q_noise, self._lowers, self._uppers)
        q = jp.concatenate([q_robot, q_cube])

        data = mjx_env.make_data(
            self._mj_model,
            qpos=q,
            ctrl=ctrl,
            qvel=self._init_qvel,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        reward, done = jp.zeros(2)
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(self._mjx_model.nu),
            "target_pos": data.site_xpos[self._target_site_id]
        }
        obs = self._get_obs(data, info)

        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        '''
        step 函数更新场景，更新6个参数
        1. data: 物理仿真数据
        2. obs: 观测(joint_pos, cube_pos, fingertip_pos, last_act)
        3. reward: 奖励(fingertip_approach, action_rate, termination)
        4. done: 是否结束(fall_termination, nans)
        5. metrics: 指标(reward/fingertip_approach, reward/action_rate, reward/termination)
        6. info: 信息(rng, step, last_act)
        '''

        # 1. 更新 data
        # 执行一次动作
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )

        # 2. 计算 obs
        obs = self._get_obs(data, state.info)

        # 3. 计算 reward
        rewards = self._get_reward(data, action, state.info, state.metrics)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        # 4. 计算 done
        done = self._get_termination(data, state.info)
        # done = jp.array(False)
        done = done.astype(reward.dtype)

        # 5. 更新 metrics
        metrics = state.metrics.copy()
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v

        # 6. 更新 info
        info = state.info.copy()
        info["step"] = info["step"] + 1
        info["last_act"] = action

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
        '''
        观测函数
        1. joint_pos: 关节位置
        2. cube_pos: 方块位置(x, y, z)
        3. fingertip_pos: 指尖位置(x, y, z)
        4. last_act: 上一个动作
        5. target_pos: 目标位置(x, y, z)
        6. distance: 目标位置与方块位置的距离
        '''
        cube_pos = data.xpos[self._cube_body_id]
        fingertip_pos = data.site_xpos[self._fingertip_site_ids].reshape(-1)
        joint_pos = data.qpos[self._all_qids]
        last_act = info["last_act"]
        target_pos = data.site_xpos[self._target_site_id]
        distance = target_pos - cube_pos
        return jp.concatenate([joint_pos, cube_pos, fingertip_pos, last_act, target_pos, distance], axis=-1)

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict, metrics: dict) -> dict:
        return {
            "fingertip_approach": self._reward_fingertip_approach(data),
            "good_finger_contact": self._reward_good_finger_contact(data),
            "position_tracking": self._reward_position_tracking(data),
            "success": self._reward_success(data),
            "action_l2": self._cost_action_l2(action),
            "action_rate_l2": self._cost_action_rate_l2(action, info["last_act"]),
            "termination": self._get_termination(data, info),
        }

    def _get_termination(self, data: mjx.Data, info: dict) -> jax.Array:
        cube_pos = data.xpos[self._cube_body_id]
        object_out_of_bound = (
            (cube_pos[0] < -1.0) | (cube_pos[0] > 1.0) |
            (cube_pos[1] < -1.0) | (cube_pos[1] > 1.0) |
            (cube_pos[2] < -0.05)  | (cube_pos[2] > 2.0)
        )

        v_limit = 5.0
        abnormal_robot = jp.any(jp.abs(data.qvel[self._all_dqids]) > 2.0 * v_limit)

        nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        return object_out_of_bound | abnormal_robot | nans

    '''定义一些reward函数'''
    def _reward_fingertip_approach(self, data: mjx.Data) -> jax.Array:
        '''
        奖励：指尖靠近物体
        '''
        cube_pos = data.xpos[self._cube_body_id]
        fingertip_pos = data.site_xpos[self._fingertip_site_ids]
        object_ee_distance = jp.max(jp.linalg.norm(fingertip_pos - cube_pos, axis=1))
        return 1 - jp.tanh(object_ee_distance / 0.15)

    def _reward_good_finger_contact(self, data: mjx.Data) -> jax.Array:
        '''
        奖励：指尖接触物体
        '''
        contact = self.get_fingertip_cube_contact(data)
        good_finger_contact = contact[0] & (contact[1] | contact[2] | contact[3] | contact[4])
        return good_finger_contact

    def _reward_position_tracking(self, data: mjx.Data) -> jax.Array:
        '''
        奖励：位置跟踪
        '''
        target_pos = data.site_xpos[self._target_site_id]
        cube_pos = data.xpos[self._cube_body_id]
        distance = jp.linalg.norm(target_pos - cube_pos)
        has_contact = jp.any(self._reward_good_finger_contact(data))
        return (1 - jp.tanh(distance / 0.4)) * has_contact

    def _reward_success(self, data: mjx.Data) -> jax.Array:
        '''
        奖励：成功
        '''
        target_pos = data.site_xpos[self._target_site_id]
        cube_pos = data.xpos[self._cube_body_id]
        distance = jp.linalg.norm(target_pos - cube_pos)
        return jp.square((1 - jp.tanh(distance / 0.1)))

    def _cost_action_l2(self, action: jax.Array) -> jax.Array:
        '''
        惩罚：动作幅度
        '''
        return jp.sum(jp.square(action))

    def _cost_action_rate_l2(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
        '''
        惩罚：动作变化率
        '''
        return jp.sum(jp.square(action - last_action))

    def render(
        self,
        trajectory,
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option=None,
        modify_scene_fns=None,
    ):
        return super().render(
            trajectory,
            height=height,
            width=width,
            camera="default" if camera is None else camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )