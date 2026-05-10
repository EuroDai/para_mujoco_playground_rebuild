# ParaNontendonFR3 Grasp 环境搭建计划（循序渐进）

## Context

用户希望在 mujoco_playground 框架下实现一个 FR3 + Parahand 的灵巧操作强化学习环境，
位置：`mujoco_playground/_src/parahand/para_nontendon_fr3/grasp.py`，
参考 IsaacLab `dexsuite/Lift` 任务的结构与奖励设计。

**当前状态**：
- XML 资产已就绪（`xmls/para_nontendon_fr3.xml`，6 DOF FR3 + 5 指 Parahand = 26 DOF，cube freejoint，touch sensor 与 framepos sensor 完备，"home" keyframe 已设置）。
- `base.py` 框架已搭，但引用了不存在的 `para_nontendon_fr3_constants` 与 menagerie 资产路径。
- `parahand/__init__.py` 注册表已就绪，但 `_src/registry.py` 不路由 parahand（已知 bug）。
- 现有 `grasp.py` 与 `parahand_params.py` 是占位代码，需要重写。

**策略**：分三大阶段，每阶段都跑通端到端训练再继续。先 reach（手指接近物体）→ 简化 Lift（接触门控位置追踪）→ 完整 Lift（特权观察 + 噪声 + 域随机化）。

---

## Stage 0 — Bootstrap（基础设施修复，1 次提交）

**目标**：让 `registry.load("ParaNontendonFR3Grasp")` 不报错，最小骨架能 reset / step。

### 任务

1. **创建** `mujoco_playground/_src/parahand/para_nontendon_fr3/para_nontendon_fr3_constants.py`：
   ```python
   from etils import epath
   from mujoco_playground._src import mjx_env
   ROOT_PATH = mjx_env.ROOT_PATH / "parahand" / "para_nontendon_fr3"
   GRASP_XML = ROOT_PATH / "xmls" / "para_nontendon_fr3.xml"
   FR3_JOINTS = ["j1","j2","j3","j4","j5","j6"]
   THUMB_JOINTS = [f"thumb_joint_{i}" for i in range(4)]
   FINGER_JOINTS_TPL = ["{prefix}_swing","{prefix}_joint_0","{prefix}_joint_1","{prefix}_joint_2"]
   FINGER_JOINTS = sum(
       ([t.format(prefix=p) for t in FINGER_JOINTS_TPL]
        for p in ["index","middle","ring","little"]), [])
   ALL_JOINTS = FR3_JOINTS + THUMB_JOINTS + FINGER_JOINTS  # 26
   FINGERTIP_SITES = ["thumb_tip","index_tip","middle_tip","ring_tip","little_tip"]
   TOUCH_SENSORS = [f"{p}_touch" for p in ["thumb","index","middle","ring","little"]]
   ```
   > 实际关节名以 XML 为准，需要先 grep XML 确认 swing/joint_0/1/2 命名一致。

2. **修复** `mujoco_playground/_src/parahand/para_nontendon_fr3/base.py`：
   - 把 `get_assets()` 简化为只加载 `xmls/*.xml`（XML 自包含，无 menagerie 依赖）。
   - 删除 `MENAGERIE_PATH / "para_nontendon_fr3"` 的引用。

3. **修复** `mujoco_playground/_src/registry.py`：
   - 在 `get_default_config` 与 `load` 中分别添加 `elif env_name in parahand.ALL_ENVS: ...` 分支。
   - 在 `get_domain_randomizer` 也加 parahand 分支。

4. **重写** `mujoco_playground/_src/parahand/para_nontendon_fr3/grasp.py` 为最小骨架：
   - `default_config()`：含 `ctrl_dt=0.02, sim_dt=0.002, episode_length=200, action_repeat=1, action_scale=0.05, impl='warp', naconmax=30*8192, naccdmax=30*8192, njmax=160`，`reward_config.scales` 暂留空 dict。
   - `class ParaNontendonFR3Grasp(base.ParaNontendonFR3Env)`：
     - `__init__` 调用 `super().__init__(consts.GRASP_XML.as_posix(), config, config_overrides)` 后 `_post_init()`。
     - `_post_init()`：缓存 `_init_q`、`_lowers/_uppers`、`_default_pose`（actuator 默认 ctrl）。
     - `reset()`：从 home keyframe 直接初始化（不随机），`mjx_env.make_data` 构造 data，info 含 `rng/step/last_act`，metrics 空 dict，obs 返回 `jp.zeros(1)` 占位。
     - `step()`：增量动作 + clip + `mjx_env.step`，reward=0, done=False。
     - `_get_obs/_get_reward/_get_termination` 留 stub。

5. **简化** `mujoco_playground/config/parahand_params.py`：
   - 把 `value_obs_key` 改为先暂用 `"state"`（Stage 3 再切到 `"privileged_state"`）。
   - 训练步数先调小（如 `num_timesteps=5_000_000`，`num_envs=1024`）便于 Stage 0/1 快速验证。

### 验证

```bash
# 单元测试：能加载、reset、step 一次
python -c "
from mujoco_playground._src import registry
import jax
env = registry.load('ParaNontendonFR3Grasp')
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, jax.numpy.zeros(env.action_size))
print('OK', state.obs.shape if hasattr(state.obs,'shape') else {k:v.shape for k,v in state.obs.items()})
"
```

---

## Stage 1 — Reach 任务（让手接近 cube）

**目标**：训练手指中心接近 cube。这是最简化的任务，用来验证整条 RL pipeline 通畅。

### 任务

1. **完善 `_post_init()`**：缓存
   - `_all_qids = mjx_env.get_qpos_ids(mj_model, ALL_JOINTS)`（26）
   - `_all_dqids = mjx_env.get_qvel_ids(mj_model, ALL_JOINTS)`
   - `_cube_qids = mjx_env.get_qpos_ids(mj_model, ["cube_freejoint"])`（7）
   - `_cube_body_id = mj_model.body("cube").id`
   - `_fingertip_site_ids = np.array([mj_model.site(n).id for n in FINGERTIP_SITES])`（5）
   - `_floor_geom_id = mj_model.geom("floor").id`

2. **`reset()`**：home keyframe + 关节微噪声（`scale=0.05` 截断到 ctrlrange），cube 不动。

3. **`_get_obs()`** 返回 dict：
   - `state` = concat([joint_angles(26), cube_pos(3), fingertip_pos_flat(15), last_act(26)]) → 70 维
   - 暂时 `privileged_state = state`（Stage 3 再分离）

4. **`_get_reward()`**：
   - `fingertip_approach`: `1 - tanh(max_finger_dist_to_cube / 0.15)`（仿 dexsuite 的 `object_ee_distance`，注意用 max 而非 mean，鼓励所有手指都靠近）
   - `action_rate`: `-sum((act-last_act)^2)`
   - `termination`: 暂时只用 NaN

5. **`reward_config.scales`**：`{fingertip_approach: 1.0, action_rate: -0.005, termination: -100.0}`

6. **`_get_termination()`**：NaN 检测 + cube_z < -0.05（掉地）。

7. **step() 模板**（参考 leap_hand/reorient.py 的 step）：
   ```
   motor_targets = clip(data.ctrl + action * action_scale, lowers, uppers)
   data = mjx_env.step(...)
   rewards = self._get_reward(...)
   reward = sum(scales[k]*v for k,v in rewards.items()) * self.dt
   info["last_act"] = action; info["step"] += 1
   metrics["reward/<k>"] = scaled_v
   ```

### 验证

- `train_jax_ppo.py --env_name=ParaNontendonFR3Grasp` 能运行至少 1M 步。
- 训练曲线中 `reward/fingertip_approach` 单调上升至 ≥ 0.7。
- 渲染 rollout（`learning/train_jax_ppo.py` 的 eval video 或自己写一个 `mediapy` 脚本）：可以肉眼看到手指逐渐合拢到 cube 上方。

---

## Stage 2 — 简化 Lift（接触门控位置追踪）

**目标**：让机器人抓起 cube 并搬到指定位置。Reset 时采样一次目标位置。

### 任务

1. **目标位置**：在 reset 中采样 `target_pos`（机器人基座系，参考 dexsuite ranges：x∈[-0.6,-0.4], y∈[-0.4,0.4], z∈[0.30,0.50]，但要考虑当前 XML 的 cube 起点 (-0.4, -0.25, 0.05)，可先用收紧的范围 z∈[0.15,0.25] 试试）。
   - 存入 `info["target_pos"]` (jax.Array shape (3,))。
   - 在 obs 中加入 `target_pos`（3）与 `target_pos - cube_pos`（3）。

2. **cube reset 随机化**：dexsuite 风格的 x∈[-0.05,0.05], y∈[-0.05,0.05] uniform offset 加在 home cube_qpos 上；orientation 暂时不随机（避免增加难度）。

3. **新 reward 项**（参考 dexsuite/Lift `RewardsCfg`）：
   - `fingers_to_object`：保留（Stage 1 的 `fingertip_approach`，可改名对齐）。
   - `position_tracking`: `(1 - tanh(||cube_pos - target_pos|| / 0.4)) * has_contact`，`has_contact` = 拇指 touch > 1 N **AND** 任意其它指 touch > 1 N。
   - `success`: `(1 - tanh(||cube_pos - target_pos|| / 0.1))^2`。
   - `good_finger_contact`: 同 `has_contact` 的布尔值。
   - `action_l2`: `sum(action^2)`（带 clamp）。
   - `action_rate_l2`: `sum((act - last_act)^2)`。
   - `early_termination`: -1 当 `abnormal_robot` 触发。
   - **scales**（直接对齐 dexsuite）：fingers_to_object=1, position_tracking=2, success=10, good_finger_contact=0.5, action_l2=-0.005, action_rate_l2=-0.005, early_termination=-1。

4. **新 termination**：
   - `object_out_of_bound`: cube 位置在 x∈(-1,1), y∈(-1,1), z∈(0.0,2.0) 之外。
   - `abnormal_robot`: `any(|qvel[all_dqids]| > 2 * v_limit)`，v_limit 暂用 5.0。
   - 把这些与 NaN 用 `|` 合并。

5. **Touch sensor 读取助手**：在 `_post_init` 中预计算 `_touch_adrs = sensor_adr[sensor("xxx_touch").id]`，`_get_touch_forces(data)` 返回 (5,) 的 jax 向量。

6. **观察更新**：
   - `state` += `target_pos(3)` + `cube_pos - target_pos(3)` + `touch_forces(5)`，约 81 维。

### 验证

- 训练 5M+ 步，`reward/success` 出现非零值，`reward/position_tracking` 上升到 0.3+。
- Eval rollout 能看到至少部分 episode 成功抓起 cube 并向 target 移动。
- 加一个简单 success rate 指标：`metrics["success_rate"] = (||cube-target|| < 0.05) & has_contact`。

---

## Stage 3 — 完整 Lift（特权观察 + 噪声 + 域随机化）

**目标**：对齐 dexsuite Lift 的鲁棒性配置。是否要 16 物体 / 点云 / history=5 留到本阶段决定。

### 任务

1. **特权观察分离**（asymmetric actor-critic）：
   - `state` = 之前的（actor 看到，可加噪声）。
   - `privileged_state` = `state` + concat([qvel(26), actuator_force(26), cube_linvel(3), cube_angvel(3), fingertip_xpos_full(15)])。
   - 修改 `parahand_params.py`：`value_obs_key="privileged_state"`。

2. **观察噪声**（参考 leap_hand/reorient.py 的 `_get_obs`）：
   - 在 `_get_obs` 内 `info["rng"], noise_rng = jax.random.split(info["rng"])`，对 joint_pos / cube_pos 加均匀噪声。
   - `noise_config.scales = {joint_pos: 0.05, cube_pos: 0.02}`。

3. **历史缓冲**（可选，按训练效果决定是否启用）：
   - `info["qpos_error_history"] = jp.zeros(history_len * 26)`，每步 `jp.roll`。
   - `state` 中加入历史误差。
   - dexsuite 用 history=5；先用 1 看效果，差再开。

4. **域随机化**（实现 `domain_randomize(model, rng)` 函数，注册到 `parahand/__init__.py` 的 `_randomizer`）：
   参考 leap_hand/reorient.py 的 `domain_randomize`：
   - 指尖摩擦：`U(0.5, 1.0)`
   - cube 质量：`*U(0.8, 1.2)`，质心偏移 `±5e-3`
   - 关节摩擦：`*U(0.5, 2.0)`
   - 关节阻尼：`*U(0.8, 1.2)`
   - 执行器 kp：`*U(0.8, 1.2)`，bias 同步更新
   - 起始关节位置：`+U(-0.05, 0.05)`

5. **可选扩展**（按需要在本阶段后续做）：
   - **多物体**：参考 dexsuite 16 个 primitive，需要修改 XML 在 reset 时切换 geom（mjx 中比较麻烦，可推迟）。
   - **点云观察**：用 256 个 FPS 采样点（可推迟）。
   - **History=5**：见上文。

### 验证

- 训练 50M+ 步，success_rate ≥ 0.6（对齐 dexsuite 论文水平更高，但作为初版可接受）。
- 在加噪声 / 改质量 / 改摩擦的测试环境中，policy 不显著退化（zero-shot 鲁棒性测试）。

---

## 关键文件清单

| 文件 | 操作 |
|---|---|
| `mujoco_playground/_src/registry.py` | Stage 0 修复 parahand 路由 |
| `mujoco_playground/_src/parahand/__init__.py` | Stage 3 注册 `_randomizer` |
| `mujoco_playground/_src/parahand/para_nontendon_fr3/__init__.py` | 不动 |
| **新建** `mujoco_playground/_src/parahand/para_nontendon_fr3/para_nontendon_fr3_constants.py` | Stage 0 |
| `mujoco_playground/_src/parahand/para_nontendon_fr3/base.py` | Stage 0 修复 `get_assets` |
| `mujoco_playground/_src/parahand/para_nontendon_fr3/grasp.py` | Stage 0/1/2/3 渐进重写 |
| `mujoco_playground/_src/parahand/para_nontendon_fr3/xmls/para_nontendon_fr3.xml` | 暂不动；Stage 3 若加多物体再改 |
| `mujoco_playground/config/parahand_params.py` | Stage 0 简化训练步数；Stage 3 切 privileged value obs |

## 复用的现成工具/函数

- `mjx_env.MjxEnv`、`mjx_env.State`、`mjx_env.make_data`、`mjx_env.step`、`mjx_env.get_qpos_ids`、`mjx_env.get_qvel_ids`、`mjx_env.update_assets`、`mjx_env.get_sensor_data`（`mujoco_playground/_src/mjx_env.py`）
- `reward.tolerance`（`mujoco_playground/_src/reward.py`）—— DM Control 风格容差函数，用于 success / lift 奖励整形。
- 模板：`mujoco_playground/_src/manipulation/leap_hand/reorient.py` —— 含 `_post_init` / `reset` / `step` / `_get_obs` (state+privileged) / `_get_reward` / `_get_termination` / `domain_randomize` 完整模板，**与本任务结构最匹配**。
- 模板：`mujoco_playground/_src/manipulation/franka_emika_panda/pick.py` —— 增量动作 + 抓物体的 panda 参考。

## 端到端验证

```bash
# Stage 0: 加载与 step 测试
python -c "from mujoco_playground._src import registry; import jax; \
e=registry.load('ParaNontendonFR3Grasp'); \
s=e.reset(jax.random.PRNGKey(0)); \
print('reset OK, obs keys:', list(s.obs.keys()) if hasattr(s.obs,'keys') else s.obs.shape); \
s=e.step(s, jax.numpy.zeros(e.action_size)); print('step OK')"

# Stage 1/2/3: 跑短训练验证 reward 上升
bash train_jax_ppo.sh   # 已有脚本，可能需要调 env_name 参数
# 或直接：
python learning/train_jax_ppo.py --env_name=ParaNontendonFR3Grasp

# 渲染策略 rollout
# 参考 reorient 的 eval video 输出，wandb run 中查看
```
