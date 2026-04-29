import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import mujoco
import mujoco.viewer
import numpy as np

ASSETS_DIR = os.environ.get(
    "SO101_DIR",
    "/home/parc/Desktop/InverseKinematics/so101-inverse-kinematics-main/so101",
)
XML_FILE = os.path.join(ASSETS_DIR, "so_101.xml")

# qpos index → controller joint name
_JOINT_MAP = [
    "shoulder_pan",   # qpos[0] = Rotation
    "shoulder_lift",  # qpos[1] = Pitch
    "elbow_flex",     # qpos[2] = Elbow
    "wrist_flex",     # qpos[3] = Wrist_Pitch
    "wrist_roll",     # qpos[4] = Wrist_Roll
    "gripper",        # qpos[5] = Jaw
]


def _qpos_to_joints(qpos) -> dict:
    return {
        _JOINT_MAP[i]: round(math.degrees(float(qpos[i])), 2)
        for i in range(min(len(qpos), len(_JOINT_MAP)))
    }


def _send_joints(controller_url: str, joints: dict) -> bool:
    payload = json.dumps({"joints": joints}).encode()
    req = urllib.request.Request(
        f"{controller_url}/api/robot/joints",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.15) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


class SO101Viewer:
    def __init__(self, xml_path: str = None, title: str = "SO101 Arm"):
        os.chdir(ASSETS_DIR)
        self.model = mujoco.MjModel.from_xml_path(xml_path or XML_FILE)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = [0, -1.57, 1.57, 0, 0, 0]
        mujoco.mj_step(self.model, self.data)
        # launch_passive returns a handle immediately so we can run our own loop
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

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

    def show(self, controller_url: str = None, rate_hz: float = 20.0):
        """
        Run the physics loop until the viewer window is closed.

        controller_url: if given, joint angles are forwarded to the real arm
                        at up to rate_hz commands per second.
        """
        min_interval = 1.0 / rate_hz
        last_send = 0.0
        last_qpos = None

        if controller_url:
            print(f"Forwarding to real arm at {controller_url} ({rate_hz:.0f} Hz)", file=sys.stderr)
            print("Tip: Ctrl+drag a body in the viewer to perturb the arm.", file=sys.stderr)
        else:
            print("Viewer running (no real-arm forwarding).", file=sys.stderr)
        print("Close the window to exit.", file=sys.stderr)

        with self.viewer as v:
            while v.is_running():
                mujoco.mj_step(self.model, self.data)
                v.sync()

                if controller_url:
                    now = time.monotonic()
                    if now - last_send < min_interval:
                        continue

                    qpos = self.data.qpos[:6].copy()
                    if last_qpos is None or any(
                        abs(qpos[i] - last_qpos[i]) > math.radians(0.5) for i in range(6)
                    ):
                        joints = _qpos_to_joints(qpos)
                        ok = _send_joints(controller_url, joints)
                        last_send = now
                        last_qpos = qpos.tolist()
                        status = "OK" if ok else "FAIL"
                        vals = "  ".join(f"{k[:6]}={v:+.1f}" for k, v in joints.items())
                        print(f"[{status}] {vals}", end="\r", flush=True)

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
    parser = argparse.ArgumentParser(description="SO101 MuJoCo viewer with optional real-arm sync")
    parser.add_argument("--controller", default=None,
                        metavar="URL",
                        help="Controller URL to forward joints to (e.g. http://127.0.0.1:5000)")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Max command rate in Hz when forwarding (default: 20)")
    args = parser.parse_args()

    print("Starting SO101Viewer...", file=sys.stderr)
    viewer = SO101Viewer()
    print("Viewer initialized", file=sys.stderr)
    viewer.show(controller_url=args.controller, rate_hz=args.rate)