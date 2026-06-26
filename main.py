import sys
from pathlib import Path

import mujoco
import mujoco.viewer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sim.mujoco_env import MujocoTwoArmEnv
from control.cooperative_fsm import CooperativeFSM


def main():
    env = MujocoTwoArmEnv()
    fsm = CooperativeFSM(env)

    print("2D cooperative control of two arms")
    print("SPACE — start / pause")
    print("R — reset")
    print("Click on MuJoCo Viewer window before pressing keys.")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=fsm.on_key) as viewer:
        while viewer.is_running():
            fsm.step()
            env.step()
            viewer.sync()


if __name__ == "__main__":
    main()