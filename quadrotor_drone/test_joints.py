import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sim.mujoco_env import MujocoTwoArmEnv


def main():
    env = MujocoTwoArmEnv()

    joints_to_test = [
        ("a_j1_R", "j1_R"),
        ("a_j2_R", "j2_R"),
        ("a_j3_R", "j3_R"),
    ]

    current_joint = 0
    t = 0.0

    print("Тест суставов правой руки")
    print("SPACE — следующий сустав")
    print("Смотри, может ли сустав пройти диапазон от -pi до +pi")

    def key_callback(keycode):
        nonlocal current_joint, t

        try:
            ch = chr(keycode)
        except Exception:
            return

        if ch == " ":
            current_joint = (current_joint + 1) % len(joints_to_test)
            t = 0.0
            print("Now testing:", joints_to_test[current_joint])

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            actuator_name, joint_name = joints_to_test[current_joint]

            # Остальные суставы держим в нуле
            for a in ["a_j1_R", "a_j2_R", "a_j3_R"]:
                env.set_ctrl(a, 0.0)

            # Тестируемый сустав плавно ходит от -pi до +pi
            q = np.pi * np.sin(t)

            env.set_ctrl(actuator_name, q)

            if int(t * 10) % 20 == 0:
                print(f"testing {joint_name}: target={q:.3f}")

            t += 0.01

            env.step()
            viewer.sync()


if __name__ == "__main__":
    main()