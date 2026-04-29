import numpy as np
import mujoco.viewer
import sys

ASSETS_DIR = "/home/parc/Desktop/InverseKinematics/so101-inverse-kinematics-main/so101"
XML_FILE = ASSETS_DIR + "/so_101.xml"


class SO101Viewer:
    def __init__(self, xml_path: str = None, title: str = "SO101 Arm"):
        import os
        os.chdir(ASSETS_DIR)
        self.model = mujoco.MjModel.from_xml_path(xml_path or XML_FILE)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = [0, -1.57, 1.57, 0, 0, 0]
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco.viewer.launch(self.model, self.data)

    def set_joint_angles(self, qpos):
        self.data.qpos[:len(qpos)] = qpos
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def get_joint_angles(self):
        return self.data.qpos.copy()

    def get_end_effector_pos(self):
        mujoco.mj_forward(self.model, self.data)
        return self.data.body("Moving_Jaw").xpos.copy()

    def render(self):
        self.viewer.sync()
        return True

    def show(self):
        print("Viewer launched. Close the window to exit.", file=sys.stderr)
        while self.viewer.is_running():
            pass
        self.close()

    def close(self):
        self.viewer.close()


class SO101RobotController:
    def __init__(self, viewer: SO101Viewer, robot_instance=None):
        self.viewer = viewer
        self.robot = robot_instance
        self.current_qpos = np.zeros(6)

    def move_to_joint_positions(self, target_qpos, steps=100):
        start = self.current_qpos.copy()
        for i in range(steps):
            t = i / steps
            interpolated = start * (1 - t) + target_qpos * t
            self.viewer.set_joint_angles(interpolated)
            self.viewer.render()
            self.current_qpos = interpolated

    def close(self):
        self.viewer.close()
        if self.robot:
            self.robot.disconnect()


if __name__ == "__main__":
    print("Starting SO101Viewer...", file=sys.stderr)
    viewer = SO101Viewer()
    print("Viewer initialized", file=sys.stderr)
    viewer.show()