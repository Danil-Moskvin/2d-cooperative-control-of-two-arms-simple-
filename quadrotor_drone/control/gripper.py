import numpy as np


def gripper_open(env, side):
    # Максимально раскрываем пальцы, чтобы они не цепляли кубик при подъезде
    env.set_ctrl(f"a_fL_{side}", 0.04)
    env.set_ctrl(f"a_fR_{side}", 0.04)


def gripper_close(env, side, s=0.035):
    s = float(np.clip(s, 0.0, 0.04))

    # Рабочая схема из старого проекта:
    # левый палец идёт внутрь:  +0.04 -> -s
    # правый палец идёт внутрь:  +0.04 -> 0.04 - s
    env.set_ctrl(f"a_fL_{side}", -s)
    env.set_ctrl(f"a_fR_{side}", 0.04 - s)