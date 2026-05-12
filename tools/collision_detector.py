import mujoco
from mujoco.mjx._src import collision_driver

model_path = '/media/rvsa/extdata/daizy/mujoco_playground/mujoco_playground/_src/parahand/para_nontendon_fr3/xmls/para_nontendon_fr3.xml'
model = mujoco.MjModel.from_xml_path(model_path)

pairs = list(collision_driver.geom_pairs(model))

print(f"模型中几何体 (Geom) 的总数: {model.ngeom}")
print(f"全量组合数量 C(n,2): {model.ngeom * (model.ngeom - 1) // 2}")
print(f"经过过滤后的潜在碰撞对数量: {len(pairs)}")
