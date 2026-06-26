import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ik_step_2d(env, q_des, dof_ids, site_name, target_xy,
               lam=1e-2, step_gain=0.05, max_joint_vel=0.5):
    dt = float(env.model.opt.timestep)

    cur_xy = env.site_pos(site_name)[:2]
    err = target_xy - cur_xy

    j_xy = env.get_site_jac_xy(site_name, dof_ids)
    dq = j_xy.T @ np.linalg.solve(j_xy @ j_xy.T + lam * np.eye(2), err)

    dq *= step_gain
    dq = np.clip(dq, -max_joint_vel * dt, max_joint_vel * dt)

    return q_des + dq


def ik_step_pose_2d(env, q_des, dof_ids, site_name, target_xy, target_angle,
                    lam=1e-2, step_gain=0.045, max_joint_vel=0.45,
                    w_angle=0.25):
    dt = float(env.model.opt.timestep)

    cur_xy = env.site_pos(site_name)[:2]
    cur_angle = env.site_angle(site_name)

    err_xy = target_xy - cur_xy
    err_angle = wrap_angle(target_angle - cur_angle)

    j = env.get_site_jac_pose_2d(site_name, dof_ids)

    err = np.array([
        err_xy[0],
        err_xy[1],
        w_angle * err_angle,
    ])

    j[2, :] *= w_angle

    dq = j.T @ np.linalg.solve(j @ j.T + lam * np.eye(3), err)

    dq *= step_gain
    dq = np.clip(dq, -max_joint_vel * dt, max_joint_vel * dt)

    return q_des + dq