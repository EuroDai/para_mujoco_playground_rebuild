# MuJoCo Playground

## 1. 配环境

### 1.1 下载源码

```bash
git clone git@github.com:EuroDai/para_mujoco_playground_rebuild.git
cd para_mujoco_playground_rebuild
```

### 1.2 配环境

```bash
conda create -n mj_playground python=3.12
conda activate mj_playground
cd para_mujoco_playground_rebuild
pip install -e .
pip install -r requirements.txt
```

### 1.3 修改 brax

打开brax库中的 `train.py`

```bash
nano /path_to_your_conda_environment/lib/python3.12/site-packages/brax/training/agents/ppo/train.py
```

在 `757-759` 行左右，注释掉原来的内容并替换

```python
  '''
  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )
  '''

  # 获取设备列表
  devices = jax.local_devices()[:local_devices_to_use]

  # 构建设备网格并指定切分策略
  mesh = Mesh(devices, ('devices',))
  sharding = NamedSharding(mesh, PartitionSpec('devices'))

  # 替代 device_put_replicated：
  # 遍历 training_state 中的所有参数，手动增加一维 (len(devices), ...) 并通过 sharding 下发
  training_state = jax.tree_util.tree_map(
      lambda x: jax.device_put(jnp.broadcast_to(x, (len(devices),) + jnp.shape(x)), sharding),
      training_state
  )
```