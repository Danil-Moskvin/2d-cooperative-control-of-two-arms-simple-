import numpy as np
import mujoco

from control.ik_2d import ik_step_2d, ik_step_pose_2d

ARM_STEP_GAIN = 0.025
ARM_MAX_JOINT_VEL = 0.10


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ArmController2D:
    def __init__(self, env, side, home_q):
        self.env = env
        self.side = side
        self.home_q = np.array(home_q, dtype=float)
        self.q_des = self.home_q.copy()

        self.joints = [
            f"j1_{side}",
            f"j2_{side}",
            f"j3_{side}",
        ]

        self.actuators = [
            f"a_j1_{side}",
            f"a_j2_{side}",
            f"a_j3_{side}",
        ]

        self.site = f"grasp_{side}"

        self.dof_ids = np.array(
            [env.joint_dof_addr(j) for j in self.joints],
            dtype=int
        )

    def reset(self):
        self.q_des = self.home_q.copy()

    def move_to_xy(self, target_xy, step_gain=ARM_STEP_GAIN, max_joint_vel=ARM_MAX_JOINT_VEL):
        self.q_des = ik_step_2d(
            self.env,
            self.q_des,
            self.dof_ids,
            self.site,
            np.array(target_xy, dtype=float),
            step_gain=step_gain,
            max_joint_vel=max_joint_vel
        )

    def move_to_pose(self, target_xy, target_angle, step_gain=ARM_STEP_GAIN, max_joint_vel=ARM_MAX_JOINT_VEL):
        self.q_des = ik_step_pose_2d(
            self.env,
            self.q_des,
            self.dof_ids,
            self.site,
            np.array(target_xy, dtype=float),
            target_angle,
            step_gain=step_gain,
            max_joint_vel=max_joint_vel
        )

    def move_home(self, rate=0.45):
        dt = float(self.env.model.opt.timestep)
        dq_lim = rate * dt
        self.q_des += np.clip(self.home_q - self.q_des, -dq_lim, dq_lim)

    def is_near_xy(self, target_xy, threshold=0.03):
        cur = self.env.site_pos(self.site)[:2]
        return np.linalg.norm(cur - np.array(target_xy, dtype=float)) < threshold

    def is_home(self, threshold=0.04):
        return np.linalg.norm(self.q_des - self.home_q) < threshold

    def apply(self):
        for joint, actuator, q in zip(self.joints, self.actuators, self.q_des):
            jid = self.env.name2id(mujoco.mjtObj.mjOBJ_JOINT, joint)

            if self.env.model.jnt_limited[jid]:
                lo, hi = self.env.model.jnt_range[jid]
                q = float(np.clip(q, lo, hi))

            self.env.set_ctrl(actuator, q)

    def move_joint_towards(self, joint_index, target, rate=0.4):
        dt = float(self.env.model.opt.timestep)
        dq_lim = rate * dt

        cur = self.q_des[joint_index]

        cur = (cur + np.pi) % (2 * np.pi) - np.pi
        target = (target + np.pi) % (2 * np.pi) - np.pi

        err = (target - cur + np.pi) % (2 * np.pi) - np.pi
        dq = np.clip(err, -dq_lim, dq_lim)

        self.q_des[joint_index] = (cur + dq + np.pi) % (2 * np.pi) - np.pi

    def sync_from_sim(self):
        self.q_des = self.env.arm_qpos(self.side).copy()