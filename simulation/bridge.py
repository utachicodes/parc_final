"""
Simulation → Real Arm Bridge
Reads qpos from the MuJoCo simulation WebSocket and forwards joint angles
to the SO-101 controller (Flask app).

Usage:
    python bridge.py [--sim-ws ws://127.0.0.1:38001] [--controller http://127.0.0.1:5000]
"""
import asyncio
import argparse
import json
import math
import os
import time
import sys
import urllib.request
import urllib.error

import websockets

# qpos index → controller joint name
JOINT_MAP = [
    "shoulder_pan",   # qpos[0] = Rotation
    "shoulder_lift",  # qpos[1] = Pitch
    "elbow_flex",     # qpos[2] = Elbow
    "wrist_flex",     # qpos[3] = Wrist_Pitch
    "wrist_roll",     # qpos[4] = Wrist_Roll
    "gripper",        # qpos[5] = Jaw
]


def qpos_to_joints(qpos: list[float]) -> dict[str, float]:
    """Convert MuJoCo qpos (radians) to controller joint dict (degrees)."""
    return {
        JOINT_MAP[i]: round(math.degrees(qpos[i]), 2)
        for i in range(min(len(qpos), len(JOINT_MAP)))
    }


def send_to_controller(controller_url: str, joints: dict[str, float]) -> bool:
    payload = json.dumps({"joints": joints}).encode()
    req = urllib.request.Request(
        f"{controller_url}/api/robot/joints",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.2) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


async def run_bridge(sim_ws: str, controller_url: str, rate_hz: float = 20.0):
    min_interval = 1.0 / rate_hz
    last_send = 0.0
    last_qpos = None

    print(f"Connecting to simulation at {sim_ws} ...")
    async with websockets.connect(sim_ws) as ws:
        print(f"Connected. Forwarding to controller at {controller_url} at {rate_hz:.0f} Hz")
        print("Press Ctrl+C to stop.\n")
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            qpos = msg.get("qpos")
            if not qpos or len(qpos) < 6:
                continue

            now = time.monotonic()
            if now - last_send < min_interval:
                continue  # rate-limit

            if last_qpos is not None:
                # Skip if nothing changed (avoid hammering servos with identical commands)
                if all(abs(qpos[i] - last_qpos[i]) < math.radians(0.5) for i in range(6)):
                    continue

            joints = qpos_to_joints(qpos)
            ok = send_to_controller(controller_url, joints)
            last_send = now
            last_qpos = list(qpos)

            status = "OK" if ok else "FAIL"
            vals = "  ".join(f"{k[:6]}={v:+.1f}" for k, v in joints.items())
            print(f"[{status}] {vals}", end="\r", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sim→Real bridge for SO-101")
    parser.add_argument("--sim-ws", default=os.environ.get("SIM_WS", "ws://127.0.0.1:38001"),
                        help="Simulation WebSocket URL")
    parser.add_argument("--controller", default=os.environ.get("CONTROLLER_URL", "http://127.0.0.1:5000"),
                        help="Controller Flask URL")
    parser.add_argument("--rate", type=float, default=float(os.environ.get("RATE_HZ", "20")),
                        help="Max command rate in Hz (default: 20)")
    args = parser.parse_args()

    try:
        asyncio.run(run_bridge(args.sim_ws, args.controller, args.rate))
    except KeyboardInterrupt:
        print("\nBridge stopped.")
    except websockets.exceptions.ConnectionClosedError:
        print("\nSimulation WebSocket closed.", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
