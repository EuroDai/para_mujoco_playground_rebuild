import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. 加载模型与数据
xml_path = "mujoco_playground/_src/parahand/para_fr3/xmls/para_fr3.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 提前获取需要监控的 geom 的内部 ID
middle_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "middle_tip")
ring_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ring_tip")

print("==================================================")
print("仿真已启动！请在弹出的 GUI 窗口中操作：")
print("  - 双击选中绿色方块 (cube)")
print("  - 按住 Ctrl + 鼠标右键 拖动方块去撞击机器人的指尖")
print("命令行将在此处实时打印碰撞力大小...")
print("==================================================")

# 3. 启动被动模式 GUI
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 只要关闭了 GUI 窗口，循环就会结束
    while viewer.is_running():
        # 步进一次物理引擎
        mujoco.mj_step(model, data)
        
        middle_total_force = np.zeros(6)
        ring_total_force = np.zeros(6)
        
        # 遍历当前帧的接触点，计算力
        for i in range(data.ncon):
            contact = data.contact[i]
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            
            if contact.geom1 == middle_tip_id or contact.geom2 == middle_tip_id:
                sign = -1.0 if contact.geom1 == middle_tip_id else 1.0
                middle_total_force += sign * c_force
                
            if contact.geom1 == ring_tip_id or contact.geom2 == ring_tip_id:
                sign = -1.0 if contact.geom1 == ring_tip_id else 1.0
                ring_total_force += sign * c_force

        # 计算力的模长 (只取前3维的线作用力，忽略力矩)
        middle_force_mag = np.linalg.norm(middle_total_force[:3])
        ring_force_mag = np.linalg.norm(ring_total_force[:3])
        
        # 【过滤输出】：为了防止终端刷屏太快看不清，我们只在产生实际接触时才 print
        # 设定一个微小的力学死区 (比如 0.001N)，过滤掉计算底噪
        if middle_force_mag > 0.001 or ring_force_mag > 0.001:
            print(f"Middle: {middle_force_mag:>6.3f} N  |  Ring: {ring_force_mag:>6.3f} N")
            
        # 同步物理状态到 GUI 渲染画面
        viewer.sync()
        
        # 保持与真实时间的同步，避免仿真跑得太快
        time.sleep(model.opt.timestep)