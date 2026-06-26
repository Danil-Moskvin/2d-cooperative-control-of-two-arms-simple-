from pathlib import Path
import numpy as np
import mujoco


class MujocoTwoArmEnv:
    def __init__(self):
        xml_path = Path(__file__).resolve().parent / "two_arm_env.xml"

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.qpos_init = self.data.qpos.copy()
        self.qvel_init = self.data.qvel.copy()

    def reset(self):
        self.data.qpos[:] = self.qpos_init
        self.data.qvel[:] = self.qvel_init
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def name2id(self, obj_type, name):
        return mujoco.mj_name2id(self.model, obj_type, name)

    def joint_qpos_addr(self, joint_name):
        jid = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[jid])

    def joint_dof_addr(self, joint_name):
        jid = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_dofadr[jid])

    def actuator_id(self, actuator_name):
        return self.name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def set_ctrl(self, actuator_name, value):
        self.data.ctrl[self.actuator_id(actuator_name)] = float(value)

    def site_pos(self, site_name):
        sid = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def body_pos(self, body_name):
        bid = self.name2id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid].copy()

    def site_angle(self, site_name):
        sid = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        rot = self.data.site_xmat[sid].reshape(3, 3)
        x_axis = rot[:, 0]
        return float(np.arctan2(x_axis[1], x_axis[0]))

    def get_site_jac_xy(self, site_name, dof_ids):
        sid = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)

        return jacp[:2, dof_ids]

    def get_site_jac_pose_2d(self, site_name, dof_ids):
        sid = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)

        return np.vstack([
            jacp[0, dof_ids],
            jacp[1, dof_ids],
            jacr[2, dof_ids],
        ])

    def cube_xy(self):
        return self.body_pos("cube")[:2].copy()

    def transfer_xy(self):
        return self.site_pos("transfer_site")[:2].copy()

    def final_xy(self):
        return self.site_pos("final_site")[:2].copy()

    def set_joint_qpos(self, joint_name, value):
        adr = self.joint_qpos_addr(joint_name)
        self.data.qpos[adr] = float(value)

    def set_arm_qpos(self, side, q):
        self.set_joint_qpos(f"j1_{side}", q[0])
        self.set_joint_qpos(f"j2_{side}", q[1])
        self.set_joint_qpos(f"j3_{side}", q[2])

    def site_x_axis_xy(self, site_name):
        sid = self.name2id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        rot = self.data.site_xmat[sid].reshape(3, 3)
        x_axis = rot[:, 0]
        v = x_axis[:2].copy()

        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.zeros(2)

        return v / n

    def arm_qpos(self, side):
        return np.array([
            self.data.qpos[self.joint_qpos_addr(f"j1_{side}")],
            self.data.qpos[self.joint_qpos_addr(f"j2_{side}")],
            self.data.qpos[self.joint_qpos_addr(f"j3_{side}")],
        ], dtype=float)