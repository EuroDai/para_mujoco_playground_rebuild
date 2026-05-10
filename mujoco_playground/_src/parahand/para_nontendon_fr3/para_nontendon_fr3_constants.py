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