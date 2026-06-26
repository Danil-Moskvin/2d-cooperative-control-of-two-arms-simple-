import numpy as np

from control.controller import ArmController2D
from control.gripper import gripper_open, gripper_close


class CooperativeFSM:
    def __init__(self, env):
        self.env = env

        self.left = ArmController2D(
            env,
            side="L",
            home_q=[-0.55, -0.85, 0.55]
        )

        self.right = ArmController2D(
            env,
            side="R",
            home_q=[0.55, 0.85, -0.55]
        )

        self.paused = True
        self.stage = 0
        self.counter = 0

        self.pre_target = None

        self.reset()

        self.right_pre_target = None
        self.right_pinch_s = 0.0

    def reset(self):
        self.env.reset()
        self.left.reset()
        self.right.reset()

        self.env.set_arm_qpos("L", self.left.home_q)
        self.env.set_arm_qpos("R", self.right.home_q)
        self.env.data.qvel[:] = 0.0

        gripper_open(self.env, "L")
        gripper_open(self.env, "R")

        self.paused = True
        self.stage = 0
        self.counter = 0

        self.pre_target = None
        self.right_pre_target = None

        print("RESET -> stage=0. Press SPACE to run.")
        self.right_pinch_s = 0.0

    def on_key(self, keycode):
        try:
            ch = chr(keycode)
        except Exception:
            return

        if ch == " ":
            self.paused = not self.paused
            print("RUN" if not self.paused else "PAUSED")

        elif ch in ("r", "R"):
            self.reset()

    def transfer_xy(self):
        return self.site_pos("transfer_site")[:2]

    def _next_stage(self, stage, message):
        self.stage = stage
        self.counter = 0
        print(message)

    def step(self):
        if not self.paused:
            self.update()

        self.left.apply()
        self.right.apply()

    def update(self):
        self.counter += 1

        cube_xy = self.env.cube_xy()

        if self.stage == 0:
            # Точка перед кубиком: пока только прицеливание
            self.pre_target = cube_xy + np.array([0.0, -0.14])

            print("stage=1 (move blue point before cube)")
            print("cube_xy:", cube_xy)
            print("pre_target:", self.pre_target)

            self.stage = 1
            self.counter = 0

        elif self.stage == 1:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            # Подводим синюю точку к позиции перед кубиком
            self.left.move_to_xy(self.pre_target)

            blue_xy = self.env.site_pos("grasp_L")[:2]
            dist = np.linalg.norm(blue_xy - self.pre_target)

            if self.counter % 100 == 0:
                print(f"stage=1 move: dist={dist:.4f}, blue={blue_xy}")

            if dist < 0.04:
                self._next_stage(1.5, "stage=1.5 (aim gripper to cube)")

        elif self.stage == 1.5:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_L")[:2]
            cube_xy = self.env.cube_xy()

            direction = cube_xy - blue_xy
            target_angle = np.arctan2(direction[1], direction[0])

            # Прицеливание: позиция остается текущей, меняем в основном ориентацию
            self.left.move_to_pose(
                blue_xy,
                target_angle=target_angle,
                step_gain=0.018,
                max_joint_vel=0.08
            )

            current_angle = self.env.site_angle("grasp_L")
            angle_error = abs((target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi)

            if self.counter % 100 == 0:
                print(
                    f"stage=1.5 aim: angle_error={angle_error:.4f}, "
                    f"target_angle={target_angle:.3f}, current={current_angle:.3f}, "
                    f"blue={blue_xy}, cube={cube_xy}"
                )

            if angle_error < 0.10:
                self._next_stage(2, "DONE: aim completed")


        elif self.stage == 2:

            gripper_open(self.env, "L")

            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_L")[:2]

            cube_xy = self.env.cube_xy()

            direction_to_cube = cube_xy - blue_xy

            dist = np.linalg.norm(direction_to_cube)

            target_angle = np.arctan2(direction_to_cube[1], direction_to_cube[0])

            # Движемся маленьким шагом в сторону кубика

            if dist > 1e-6:
                direction_to_cube = direction_to_cube / dist

            step_len = 0.0010

            target_xy = blue_xy + direction_to_cube * step_len

            # Одновременно двигаемся вперёд и продолжаем прицеливаться

            self.left.move_to_pose(

                target_xy,

                target_angle=target_angle,

                step_gain=0.018,

                max_joint_vel=0.07

            )

            current_angle = self.env.site_angle("grasp_L")

            angle_error = abs((target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi)

            if self.counter % 100 == 0:
                print(

                    f"stage=2 move+aim: dist={dist:.4f}, "

                    f"angle_error={angle_error:.4f}, "

                    f"blue={blue_xy}, cube={cube_xy}"

                )

            if dist < 0.005:
                self._next_stage(3, "stage=3 (close gripper)")

        elif self.stage == 3:
            gripper_close(self.env, "L", s=0.035)
            gripper_open(self.env, "R")

            if self.counter % 100 == 0:
                print("stage=3 closing left gripper")

            if self.counter > 500:
                self._next_stage(4, "stage=4 (move cube to transfer point)")

        elif self.stage == 4:

            gripper_close(self.env, "L", s=0.035)
            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_L")[:2]
            transfer_xy = self.env.transfer_xy()

            direction = transfer_xy - blue_xy
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                direction = direction / dist

            target_angle = np.arctan2(direction[1], direction[0])

            step_len = 0.0010
            target_xy = blue_xy + direction * step_len

            self.left.move_to_pose(
                target_xy,
                target_angle=target_angle,
                step_gain=0.018,
                max_joint_vel=0.08
            )

            if self.counter % 100 == 0:
                print(
                    f"stage=4 move to transfer: "
                    f"dist={dist:.4f}, "
                    f"blue={blue_xy}, "
                    f"transfer={transfer_xy}"
                )

            if dist < 0.005:
                self._next_stage(5, "stage=5 (release cube)")

        elif self.stage == 5:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            if self.counter % 100 == 0:
                print("stage=5 releasing cube")

            if self.counter > 500:
                self._next_stage(6, "stage=6 (move gripper backward)")

        elif self.stage == 6:

            gripper_open(self.env, "L")

            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_L")[:2]

            # Направление назад — противоположно направлению кисти

            backward = -self.env.site_x_axis_xy("grasp_L")

            step_len = 0.0012

            target_xy = blue_xy + backward * step_len

            self.left.move_to_xy(self.pre_target)

            if self.counter % 100 == 0:
                print("stage=6 moving gripper backward")

            # Едем назад немного времени, чтобы кисть вышла из зоны кубика

            if self.counter > 22800:
                self._next_stage(6.5, "stage=6.5 (park left arm)")

        elif self.stage == 6.5:

            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            self.left.move_home(rate=0.45)

            if self.counter % 100 == 0:
                print("stage=6.5 parking left arm")

            if self.left.is_home():
                self._next_stage(7, "stage=7 (move right blue point before cube)")

        elif self.stage == 7:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            cube_xy = self.env.cube_xy()

            if self.right_pre_target is None:
                self.right_pre_target = cube_xy + np.array([0.0, -0.14])
                print("right_pre_target:", self.right_pre_target)

            self.right.move_to_xy(self.right_pre_target)

            blue_xy = self.env.site_pos("grasp_R")[:2]
            dist = np.linalg.norm(blue_xy - self.right_pre_target)

            if self.counter % 100 == 0:
                print(f"stage=7 move right: dist={dist:.4f}, blue={blue_xy}")

            if dist < 0.04:
                self._next_stage(7.5, "stage=7.5 (aim right gripper to cube)")


        elif self.stage == 7.5:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_R")[:2]
            cube_xy = self.env.cube_xy()

            direction = cube_xy - blue_xy
            target_angle = np.arctan2(direction[1], direction[0])

            self.right.move_to_pose(
                blue_xy,
                target_angle=target_angle
            )

            current_angle = self.env.site_angle("grasp_R")
            angle_error = abs((target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi)

            if self.counter % 100 == 0:
                print(
                    f"stage=7.5 aim right: angle_error={angle_error:.4f}, "
                    f"target_angle={target_angle:.3f}, current={current_angle:.3f}"
                )

            if angle_error < 0.10:
                self._next_stage(8, "stage=8 (right move to cube with aiming)")

        elif self.stage == 8:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_R")[:2]
            cube_xy = self.env.cube_xy()

            direction_to_cube = cube_xy - blue_xy
            dist = np.linalg.norm(direction_to_cube)

            target_angle = np.arctan2(direction_to_cube[1], direction_to_cube[0])

            if dist > 1e-6:
                direction_to_cube = direction_to_cube / dist

            step_len = 0.0008
            target_xy = blue_xy + direction_to_cube * step_len

            self.right.move_to_pose(
                target_xy,
                target_angle=target_angle
            )

            current_angle = self.env.site_angle("grasp_R")
            angle_error = abs((target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi)

            if self.counter % 100 == 0:
                print(
                    f"stage=8 right move+aim: dist={dist:.4f}, "
                    f"angle_error={angle_error:.4f}, "
                    f"blue={blue_xy}, cube={cube_xy}"
                )

            if dist < 0.005:
                self.right_pinch_s = 0.0
                self._next_stage(9, "stage=9 (right close gripper)")



        elif self.stage == 9:

            gripper_open(self.env, "L")

            self.right_pinch_s = min(

                self.right_pinch_s + 0.00008,

                0.030

            )

            gripper_close(self.env, "R", s=self.right_pinch_s)

            if self.counter % 100 == 0:
                print(f"stage=9 closing right gripper s={self.right_pinch_s:.4f}")

            if self.right_pinch_s >= 0.030 and self.counter > 700:
                self._next_stage(9.5, "stage=9.5 (settle before final move)")

        elif self.stage == 9.5:
            gripper_open(self.env, "L")
            gripper_close(self.env, "R", s=0.035)

            if self.counter % 100 == 0:
                print("stage=9.5 settle before final move")

            if self.counter > 700:
                self.right.sync_from_sim()
                self._next_stage(10, "stage=10 (slow start)")




        elif self.stage == 10:

            gripper_open(self.env, "L")

            gripper_close(self.env, "R", s=0.050)

            blue_xy = self.env.site_pos("grasp_R")[:2]

            cube_xy = self.env.cube_xy()

            final_xy = self.env.final_xy()

            direction = final_xy - blue_xy

            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                direction = direction / dist

            step_len = 0.0015

            target_xy = blue_xy + direction * step_len

            self.right.move_to_xy(

                target_xy,

                step_gain=0.025,

                max_joint_vel=0.01

            )

            err = np.linalg.norm(cube_xy - blue_xy)

            if self.counter % 100 == 0:
                print(

                    "stage=10 move to final",

                    "dist=", round(dist, 4),

                    "err=", round(err, 4),

                    "cube=", np.round(cube_xy, 3),

                    "blue=", np.round(blue_xy, 3)

                )

            if dist < 0.005:
                self._next_stage(11, "DONE: cube reached final point")

            cube_xy = self.env.cube_xy()
            blue_xy = self.env.site_pos("grasp_R")[:2]
            final_xy = self.env.final_xy()

            cube_dist = np.linalg.norm(cube_xy - final_xy)
            hold_err = np.linalg.norm(cube_xy - blue_xy)

            if dist < 0.005:
                self._next_stage(11, "stage=11 (release cube)")

        elif self.stage == 11:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            if self.counter % 100 == 0:
                print("stage=11 release cube")

            if self.counter > 500:
                self.right.sync_from_sim()
                self._next_stage(12, "stage=12 (move right gripper backward)")


        elif self.stage == 12:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            blue_xy = self.env.site_pos("grasp_R")[:2]

            backward = -self.env.site_x_axis_xy("grasp_R")
            step_len = 0.0010
            target_xy = blue_xy + backward * step_len

            self.right.move_to_xy(
                target_xy,
                step_gain=0.015,
                max_joint_vel=0.05
            )

            if self.counter % 100 == 0:
                print("stage=12 moving right gripper backward")

            if self.counter > 3000:
                self.right.sync_from_sim()
                self._next_stage(13, "stage=13 (return right arm home)")


        elif self.stage == 13:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

            self.right.move_home(rate=0.35)

            if self.counter % 100 == 0:
                print("stage=13 returning right arm home")

            if self.right.is_home():
                self._next_stage(14, "DONE")


        elif self.stage == 14:
            gripper_open(self.env, "L")
            gripper_open(self.env, "R")

