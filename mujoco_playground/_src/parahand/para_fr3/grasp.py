from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
# import os
# import time
from ml_collections import config_dict
from mujoco import mjx

# 引入基础环境类
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.parahand.para_fr3 import base as para_fr3_base
from mujoco_playground._src.parahand.para_fr3 import para_fr3_constants as consts


def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,        # 策略控制频率 50Hz
        sim_dt=0.002,        # 底层物理仿真频率 500Hz
        episode_length=512,  # 每个回合最大步数
        action_repeat=1,
        action_scale_arm=0.02,    # 增量动作的缩放比例
        action_scale_hand=0.03,
        action_scale_tendon=0.0005,
        v_limit_arm=6.4,
        v_limit_hand=10,
        impl='warp', # 默认用warp，
        naconmax=30 * 4096,
        # naccdmax=240*8192,
        njmax=1500,
        history_len=2,
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
        ),
        contact_force_threshold = 0.5,
        num_points = 64,
        pointcloud_pool_points = 256,
    )
    return config


class ParaFR3Grasp(para_fr3_base.ParaFR3Env):
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
        self._arm_act_ids = np.array([self._mj_model.actuator(n).id for n in consts.ARM_JOINTS])
        self._hand_act_ids = np.array([self._mj_model.actuator(n).id for n in consts.HAND_JOINTS])
        self._finger_tendon_act_ids = np.array([self._mj_model.actuator(n).id for n in consts.FINGER_TENDONS])
        self._index_swing_act_id = self._mj_model.actuator("index_swing").id
        self._middle_swing_act_id = self._mj_model.actuator("middle_swing").id
        self._ring_swing_act_id = self._mj_model.actuator("ring_swing").id
        self._little_swing_act_id = self._mj_model.actuator("little_swing").id
        self._finger_tendon_ids = np.array([self._mj_model.tendon(n).id for n in consts.FINGER_TENDONS])
        self._all_joint_ids = np.array([self._mj_model.joint(n).id for n in consts.ALL_JOINTS])
        self._joint_lowers = self._mj_model.jnt_range[self._all_joint_ids, 0]
        self._joint_uppers = self._mj_model.jnt_range[self._all_joint_ids, 1]
        self._arm_qids = mjx_env.get_qpos_ids(self.mj_model, consts.ARM_JOINTS)
        self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.HAND_JOINTS)
        self._finger_passive_joints_qids = mjx_env.get_qpos_ids(self.mj_model, consts.FINGER_PASSIVE_JOINTS)
        self._arm_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.ARM_JOINTS)
        self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.HAND_JOINTS)
        self._finger_passive_joints_dqids = mjx_env.get_qvel_ids(
            self.mj_model, consts.FINGER_PASSIVE_JOINTS
        )
        self._all_hand_dqids = mjx_env.get_qvel_ids(
            self.mj_model, consts.HAND_JOINTS + consts.FINGER_PASSIVE_JOINTS
        )
        self._all_qids = mjx_env.get_qpos_ids(self.mj_model, consts.ALL_JOINTS)
        self._all_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.ALL_JOINTS)
        self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
        self._cube_body_id = self._mj_model.body("cube").id
        self._fingertip_tacs_ids = np.array([self._mj_model.geom(n).id for n in consts.FINGERTIP_TACS])
        self._fingertip_site_ids = np.array([self._mj_model.site(n).id for n in consts.FINGERTIP_TIPS])
        self._fingertip_body_ids = np.array([
            self._mj_model.site_bodyid[site_id] for site_id in self._fingertip_site_ids
        ])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._target_site_id = self._mj_model.site(consts.TARGET_SITE).id
        self._default_pose = self._init_q[self._all_qids]
        self._default_cube_pose = self._init_q[self._cube_qids]
        self._cube_pointcloud_pool = self._sample_box_pointcloud(
            num_points=self._config.pointcloud_pool_points,
            box_geom_name="cube",
        )

    def _apply_swing_ctrl_constraints(self, ctrl: jax.Array) -> jax.Array:
        middle_swing_lower = jp.maximum(
            ctrl[self._index_swing_act_id],
            self._lowers[self._middle_swing_act_id],
        )
        middle_swing_upper = jp.minimum(
            ctrl[self._little_swing_act_id],
            self._uppers[self._middle_swing_act_id],
        )
        ctrl = ctrl.at[self._middle_swing_act_id].set(
            jp.clip(
                ctrl[self._middle_swing_act_id],
                middle_swing_lower,
                middle_swing_upper,
            )
        )

        ring_swing_lower = jp.maximum(
            ctrl[self._middle_swing_act_id],
            self._lowers[self._ring_swing_act_id],
        )
        ring_swing_upper = jp.minimum(
            ctrl[self._little_swing_act_id],
            self._uppers[self._ring_swing_act_id],
        )
        ctrl = ctrl.at[self._ring_swing_act_id].set(
            jp.clip(
                ctrl[self._ring_swing_act_id],
                ring_swing_lower,
                ring_swing_upper,
            )
        )
        return ctrl


    def reset(self, rng: jax.Array) -> State:
        (
            rng,
            arm_q_rng,
            arm_ctrl_rng,
            hand_ctrl_rng,
            tendon_ctrl_rng,
            cube_pos_rng,
            cube_quat_rng,
        ) = jax.random.split(rng, 7)

        arm_q_noise = jax.random.normal(arm_q_rng, (len(self._arm_qids),))
        q_robot = self._default_pose.at[:len(self._arm_qids)].add(0.05 * arm_q_noise)

        arm_ctrl_noise = jax.random.normal(arm_ctrl_rng, (len(self._arm_act_ids),))
        ctrl = self._init_ctrl.at[self._arm_act_ids].add(0.05 * arm_ctrl_noise)
        hand_ctrl = jax.random.uniform(
            hand_ctrl_rng,
            (len(self._hand_act_ids),),
            minval=self._lowers[self._hand_act_ids],
            maxval=self._uppers[self._hand_act_ids],
        )
        tendon_ctrl = jax.random.uniform(
            tendon_ctrl_rng,
            (len(self._finger_tendon_act_ids),),
            minval=self._lowers[self._finger_tendon_act_ids],
            maxval=self._uppers[self._finger_tendon_act_ids],
        )
        ctrl = ctrl.at[self._hand_act_ids].set(hand_ctrl)
        ctrl = ctrl.at[self._finger_tendon_act_ids].set(tendon_ctrl)
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        ctrl = self._apply_swing_ctrl_constraints(ctrl)

        q_robot = q_robot.at[
            len(self._arm_qids):len(self._arm_qids) + len(self._hand_qids)
        ].set(ctrl[self._hand_act_ids])
        q_robot = jp.clip(q_robot, self._joint_lowers, self._joint_uppers)

        q_cube = self._default_cube_pose.copy()
        q_cube = q_cube.at[:3].set(
            self._default_cube_pose[:3]
            + jax.random.uniform(
                cube_pos_rng,
                (3,),
                minval=jp.array([-0.1, -0.1, 0.0]),
                maxval=jp.array([0.1, 0.1, 0.0]),
            )
        )

        q_cube = q_cube.at[3:].set(para_fr3_base.uniform_quat(cube_quat_rng))
        # q_cube = self._default_cube_pose.copy()

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

        settle_steps = 10
        data = jax.lax.fori_loop(
            0,
            settle_steps,
            lambda _, d: mjx_env.step(self.mjx_model, d, ctrl, self.n_substeps),
            data,
        )
        reward, done = jp.zeros(2)
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        for name in consts.ARM_JOINTS + consts.HAND_JOINTS + consts.FINGER_TENDONS:
            metrics[f"action/{name}"] = jp.zeros(())

        for name in consts.FINGERTIP_TACS:
            metrics[f"fingertip_force/{name}"] = jp.zeros(())
        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(self._mjx_model.nu),
            "target_pos": data.site_xpos[self._target_site_id],
        }
        contact_data = self._get_contact_data(data)
        fingertip_force = self.get_fingertip_cube_contact(data, contact_data)
        single_obs, single_privileged_obs = self._get_single_obs(
            data, info, fingertip_force
        )
        info["obs_history"] = jp.zeros(
            self._config.history_len * single_obs.size, dtype=single_obs.dtype
        )
        info["privileged_obs_history"] = jp.zeros(
            self._config.history_len * single_privileged_obs.size,
            dtype=single_privileged_obs.dtype,
        )
        obs = self._get_obs(data, info, fingertip_force)

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
        arm_n = len(self._arm_act_ids)
        hand_n = len(self._hand_act_ids)
        tendon_n = len(self._finger_tendon_act_ids)
        effective_action = action
        delta_arm = effective_action[:arm_n] * self._config.action_scale_arm
        delta_hand = effective_action[arm_n:arm_n+hand_n] * self._config.action_scale_hand
        delta_tendons = effective_action[arm_n+hand_n:arm_n+hand_n+tendon_n] * self._config.action_scale_tendon

        ctrl = state.data.ctrl
        ctrl = ctrl.at[self._arm_act_ids].add(delta_arm)
        ctrl = ctrl.at[self._hand_act_ids].add(delta_hand)
        ctrl = ctrl.at[self._finger_tendon_act_ids].add(delta_tendons)
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        ctrl = self._apply_swing_ctrl_constraints(ctrl)

        data = mjx_env.step(
            self.mjx_model, state.data, ctrl, self.n_substeps
        )
        
        contact_data = self._get_contact_data(data)
        fingertip_force = self.get_fingertip_cube_contact(data, contact_data)
        arm_floor_collision = self.has_geom_floor_contact(
            data, ("forearm_collision_1", "wrist2_collision_2"), contact_data
        )

        # 4. 计算 done
        done = self._get_termination(
            data,
            state.info,
            arm_floor_collision=arm_floor_collision,
        )

        # 2. 计算 obs
        obs = self._get_obs(data, state.info, fingertip_force)

        # 3. 计算 reward
        def _termination_only_rewards(_):
            rewards = {
                k: jp.zeros((), dtype=data.qpos.dtype)
                for k in self._config.reward_config.scales.keys()
            }
            rewards["termination"] = jp.asarray(
                self._config.reward_config.scales["termination"], dtype=data.qpos.dtype
            )
            return rewards

        def _normal_rewards(_):
            rewards = self._get_reward(
                data, effective_action, state.info, state.metrics, done, fingertip_force
            )
            return {
                k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
            }

        rewards = jax.lax.cond(
            done, _termination_only_rewards, _normal_rewards, operand=None
        )
        reward = sum(rewards.values()) * self.dt
        done = done.astype(reward.dtype)

        # 5. 更新 metrics
        metrics = state.metrics.copy()
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v

        fingertip_force_norm = jp.linalg.norm(fingertip_force, axis=-1)
        for name, force in zip(consts.FINGERTIP_TACS, fingertip_force_norm):
            metrics[f"fingertip_force/{name}"] = force

        for name, a in zip(
            consts.ARM_JOINTS + consts.HAND_JOINTS + consts.FINGER_TENDONS,
            effective_action,
        ):
            metrics[f"action/{name}"] = a

        # 6. 更新 info
        info = state.info.copy()
        info["step"] = info["step"] + 1
        info["last_act"] = jp.where(
            done > 0,
            jp.zeros_like(effective_action),
            effective_action,
        )
        info["obs_history"] = jp.where(
            done > 0,
            jp.zeros_like(info["obs_history"]),
            info["obs_history"],
        )
        info["privileged_obs_history"] = jp.where(
            done > 0,
            jp.zeros_like(info["privileged_obs_history"]),
            info["privileged_obs_history"],
        )

        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info
        )
        return state

    def _get_obs(
        self, data: mjx.Data, info: dict, fingertip_force: jax.Array
    ) -> mjx_env.Observation:
        state, privileged_state = self._get_single_obs(data, info, fingertip_force)
        state_history = jp.roll(info["obs_history"], state.size).at[:state.size].set(state)
        privileged_state_history = jp.roll(
            info["privileged_obs_history"], privileged_state.size
        ).at[:privileged_state.size].set(privileged_state)
        info["obs_history"] = state_history
        info["privileged_obs_history"] = privileged_state_history
        return {
            "state": state_history,
            "privileged_state": privileged_state_history,
        }

    def _get_single_obs(
        self, data: mjx.Data, info: dict, fingertip_force: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        '''
        观测函数
        policy:
        1. cube_pose:           方块完整位姿(x, y, z, qw, qx, qy, qz)
        2. target_pos:          目标位置(x, y, z)
        3. last_act:            上一个动作
        proprio:
        1. joint_pos:           关节角度
        2. joint_vel:           关节速度
        3. 指尖状态：
            fingertip_poses:    指尖位姿
            fingertip_vels:     指尖线速度、角速度
        4. fingertip_force:     指尖力
        perception:
        1. cube_pointcloud:     方块点云
        '''
        # policy
        target_pos = data.site_xpos[self._target_site_id]
        last_act = info["last_act"]
        policy_obs = jp.concatenate([target_pos, last_act])

        # proprio
        active_joint_pos = jp.concatenate([
            data.qpos[self._arm_qids],
            data.qpos[self._hand_qids],
        ])
        passive_joint_pos = data.qpos[self._finger_passive_joints_qids]
        tendon_pos = jp.clip(data.ten_length[self._finger_tendon_ids], -2.0, 2.0)
        active_joint_vel = jp.concatenate([
            data.qvel[self._arm_dqids],
            data.qvel[self._hand_dqids],
        ])
        passive_joint_vel = data.qvel[self._finger_passive_joints_dqids]
        tendon_vel = jp.clip(data.ten_velocity[self._finger_tendon_ids], -2.0, 2.0)
        fingertip_poses, fingertip_vels = self.get_fingertip_kinematics(
            data,
            jp.asarray(self._fingertip_site_ids),
            jp.asarray(self._fingertip_body_ids),
        )
        fingertip_poses = jp.clip(
            fingertip_poses.reshape(-1),
            -2.0,
            2.0,
        )
        fingertip_vels = jp.clip(
            fingertip_vels.reshape(-1),
            -2.0,
            2.0,
        )
        fingertip_force = jp.clip(
            fingertip_force.reshape(-1),
            -20.0,
            20.0,
        )
        proprio_obs = jp.concatenate([
            active_joint_pos,
            tendon_pos,
            active_joint_vel,
            tendon_vel,
            fingertip_force,
        ])

        # perception
        cube_pointcloud = self.get_box_pointcloud(
            data,
            num_points=self._config.pointcloud_pool_points,
            box_geom_name="cube",
            pointcloud=self._cube_pointcloud_pool,
        )
        rng, pointcloud_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        point_ids = jax.random.permutation(
            pointcloud_rng, cube_pointcloud.shape[0]
        )[:self._config.num_points]
        cube_pointcloud = cube_pointcloud[point_ids]
        sort_ids = jp.lexsort(
            (cube_pointcloud[:, 2], cube_pointcloud[:, 1], cube_pointcloud[:, 0])
        )
        cube_pointcloud = jp.clip(
            cube_pointcloud[sort_ids].reshape(-1),
            -2.0,
            2.0,
        )

        privileged = jp.concatenate([
            passive_joint_pos, 
            passive_joint_vel,
            fingertip_poses,
            fingertip_vels,
        ], axis=-1)

        state = jp.concatenate([policy_obs, proprio_obs, cube_pointcloud], axis=-1)
        privileged_state = jp.concatenate([state, privileged], axis=-1)
        return state, privileged_state

    def _get_reward(
        self, data: mjx.Data,
        action: jax.Array, info: dict,
        metrics: dict,
        done: jax.Array,
        fingertip_force: jax.Array,
    ) -> dict:
        del metrics
        return {
            "fingertip_approach": self._reward_fingertip_approach(data),
            "good_finger_contact": self._reward_good_finger_contact(fingertip_force),
            "position_tracking": self._reward_position_tracking(data, fingertip_force),
            "success": self._reward_success(data),
            "action_l2": self._cost_action_l2(action),
            "action_rate_l2": self._cost_action_rate_l2(action, info["last_act"]),
            "termination": jp.asarray(done, dtype=data.qpos.dtype),
        }

    def _get_termination(
        self,
        data: mjx.Data,
        info: dict,
        arm_floor_collision: Optional[jax.Array] = None,
    ) -> jax.Array:
        del info
        cube_pos = data.xpos[self._cube_body_id]
        object_out_of_bound = (
            (cube_pos[0] < -1.0) | (cube_pos[0] > 1.0) |
            (cube_pos[1] < -1.0) | (cube_pos[1] > 1.0) |
            (cube_pos[2] < 0.0)  | (cube_pos[2] > 2.0)
        )

        abnormal_arm = jp.any(jp.abs(data.qvel[self._arm_dqids]) > self._config.v_limit_arm)
        abnormal_hand = jp.any(jp.abs(data.qvel[self._all_hand_dqids]) > self._config.v_limit_hand)
        abnormal_robot = abnormal_arm | abnormal_hand

        if arm_floor_collision is None:
            arm_floor_collision = self.has_geom_floor_contact(
                data, ("forearm_collision_1", "wrist2_collision_2")
            )

        nans = (
            jp.any(jp.isnan(data.qpos)) |
            jp.any(jp.isnan(data.qvel)) |
            jp.any(jp.isnan(data.xpos)) |
            jp.any(jp.isnan(data.site_xpos))
        )
        return (
            object_out_of_bound
            | abnormal_robot
            | arm_floor_collision
            | nans
        )

    '''定义一些reward函数'''
    def _reward_fingertip_approach(self, data: mjx.Data) -> jax.Array:
        '''
        奖励：指尖靠近物体
        '''
        cube_pos = data.xpos[self._cube_body_id]
        fingertip_pos = data.site_xpos[self._fingertip_site_ids]
        object_ee_distance = jp.max(jp.linalg.norm(fingertip_pos - cube_pos, axis=1))
        return 1 - jp.tanh(object_ee_distance / 0.15)

    def _reward_good_finger_contact(self, fingertip_force: jax.Array) -> jax.Array:
        '''
        奖励：指尖接触物体
        '''
        contact_force_norm = jp.linalg.norm(fingertip_force, axis=-1)
        thumb_force = contact_force_norm[0]
        index_force = contact_force_norm[1]
        middle_force = contact_force_norm[2]
        ring_force = contact_force_norm[3]
        little_force = contact_force_norm[4]
        good_finger_contact = (
            (thumb_force > self._config.contact_force_threshold) &
            (
                (index_force > self._config.contact_force_threshold) |
                (middle_force > self._config.contact_force_threshold) |
                (ring_force > self._config.contact_force_threshold) |
                (little_force > self._config.contact_force_threshold)
            )
        )
        return good_finger_contact

    def _reward_position_tracking(
        self, data: mjx.Data, fingertip_force: jax.Array
    ) -> jax.Array:
        '''
        奖励：位置跟踪
        '''
        target_pos = data.site_xpos[self._target_site_id]
        cube_pos = data.xpos[self._cube_body_id]
        distance = jp.linalg.norm(target_pos - cube_pos)
        has_contact = jp.any(self._reward_good_finger_contact(fingertip_force))
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

    def _cost_action_rate_l2(
        self, 
        action: jax.Array, 
        last_action: jax.Array
    ) -> jax.Array:
        '''
        惩罚：动作变化率
        '''
        return jp.sum(jp.square(action - last_action))

