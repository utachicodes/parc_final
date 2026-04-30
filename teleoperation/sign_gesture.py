import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import math
import argparse
import sys

# ── Depth Anything V2 ─────────────────────────────────────────────────────────
# Make sure the Depth-Anything-V2 repo is on your PYTHONPATH:
#   git clone https://github.com/DepthAnything/Depth-Anything-V2
#   pip install -e Depth-Anything-V2
# Add the local repo to path
sys.path.append('Depth-Anything-V2-main')

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_AVAILABLE = True
except ImportError:
    print("[WARN] depth_anything_v2 not found in 'Depth-Anything-V2-main' – depth features disabled.")
    DEPTH_AVAILABLE = False

# ── Try importing the Feetech / SCServo SDK ───────────────────────────────────
SERVO_SDK = None
try:
    from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS, SCS_LOBYTE, SCS_HIBYTE
    SERVO_SDK = "scservo"
except ImportError:
    try:
        from feetech_servo_sdk import PortHandler, PacketHandler
        def SCS_LOBYTE(w): return int(w & 0xFF)
        def SCS_HIBYTE(w): return int((w >> 8) & 0xFF)
        SERVO_SDK = "feetech"
    except ImportError:
        SERVO_SDK = None

# ── Configuration ─────────────────────────────────────────────────────────────
import os
DEFAULT_PORT  = os.environ.get('TELEOP_PORT', 'COM6')
DEFAULT_BAUD  = int(os.environ.get('TELEOP_BAUD', '1000000'))
DEPTH_MODEL_PATH = "depth_anything_v2_vitb.pth"

# SO-101 servo IDs
SERVO_IDS = {
    "shoulder_pan":  1,
    "shoulder_tilt": 2,
    "elbow":         3,
    "wrist_flex":    4,
    "wrist_rot":     5,
    "gripper":       6,
}

# Safe position limits (raw servo units, 0-4095)
LIMITS = {
    "shoulder_pan":  (1024, 3072),
    "shoulder_tilt": (1024, 3072),
    "elbow":         (1024, 3072),
    "wrist_flex":    (1024, 3072),
    "wrist_rot":     (1024, 3072),
    "gripper":       (1800, 3000),
}

# Depth-to-elbow mapping (metric, in metres)
# Depth-to-elbow mapping (Relative scale 0.0 to 1.0)
DEPTH_NEAR   = 0.85   # Hand is close (large relative value) → elbow UP
DEPTH_FAR    = 0.20   # Hand is far (small relative value)  → elbow DOWN

STEP        = 30
FPS_TARGET  = 15
ADDR_GOAL   = 42
ADDR_TORQUE = 40

# Smoothing factor (0.0 to 1.0) - Lower is smoother but slower
SMOOTH_FACTOR = 0.35 
# Deadband (raw units) - Ignore movements smaller than this to prevent jitter
DEADBAND = 8

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Depth model setup ─────────────────────────────────────────────────────────
def load_depth_model(model_path: str):
    """Load Depth Anything V2 model (S or B)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect encoder based on filename
    if 'vits' in model_path.lower():
        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        print("[INFO] Using ViT-Small architecture")
    elif 'vitb' in model_path.lower():
        cfg = {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}
        print("[INFO] Using ViT-Base architecture")
    else:
        # Default to small if unknown
        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        print("[WARN] Unknown model type in filename, defaulting to ViT-S")

    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"[OK]  Depth Anything V2 loaded on {device}")
    return model

def get_depth_at(depth_map: np.ndarray, cx: int, cy: int, patch: int = 15) -> float:
    """
    Sample depth at (cx, cy) using a larger patch and median filter.
    Returns depth in metres (if using metric model) or relative scale.
    """
    h, w = depth_map.shape[:2]
    # Use a larger 15x15 patch for better stability
    x0, x1 = max(0, cx - patch), min(w, cx + patch + 1)
    y0, y1 = max(0, cy - patch), min(h, cy + patch + 1)
    region = depth_map[y0:y1, x0:x1]
    if region.size == 0: return 0.0
    return float(np.median(region))

def depth_to_elbow_pos(depth_m: float) -> int:
    """Map metric depth to an elbow servo position."""
    lo, hi = LIMITS["elbow"]
    # Clamp depth into [NEAR, FAR]
    t = np.clip((depth_m - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0.0, 1.0)
    # Near hand → elbow high (hi), far hand → elbow low (lo)
    return int(hi - t * (hi - lo))

# ── Servo controller ──────────────────────────────────────────────────────────
class ServoController:
    def __init__(self, port_name=DEFAULT_PORT, baud_rate=DEFAULT_BAUD, simulate=False):
        self.positions = {k: 2048 for k in SERVO_IDS}
        self.connected = False
        self.port_name = port_name
        self.simulate = simulate or (SERVO_SDK is None)

        if not self.simulate:
            try:
                self.port = PortHandler(port_name)
                self.handler = PacketHandler(0)
                if self.port.openPort() and self.port.setBaudRate(baud_rate):
                    self.connected = True
                    print(f"[OK]  Connected to {port_name} at {baud_rate} bps")
                    self._enable_torque()
                else:
                    print(f"[ERR] Failed to open {port_name}. Falling back to SIMULATION.")
                    self.simulate = True
            except Exception as e:
                print(f"[ERR] Serial error: {e}. Falling back to SIMULATION.")
                self.simulate = True

        if self.simulate:
            print("[INFO] Running in SIMULATION mode.")

    def _write(self, servo_id, position):
        if not self.connected or self.simulate:
            return
        self.handler.write2ByteTxRx(self.port, servo_id, ADDR_GOAL, position)

    def _enable_torque(self):
        if not self.connected or self.simulate:
            return
        for sid in SERVO_IDS.values():
            self.handler.write1ByteTxRx(self.port, sid, ADDR_TORQUE, 1)

    def disable_torque(self):
        if not self.connected or self.simulate:
            return
        for sid in SERVO_IDS.values():
            self.handler.write1ByteTxRx(self.port, sid, ADDR_TORQUE, 0)

    def move(self, joint: str, delta: int):
        lo, hi = LIMITS[joint]
        new_pos = max(lo, min(hi, self.positions[joint] + delta))
        self.positions[joint] = new_pos
        self._write(SERVO_IDS[joint], new_pos)

    def set_pos(self, joint: str, target_pos: int):
        lo, hi = LIMITS[joint]
        target_pos = max(lo, min(hi, target_pos))
        
        # ── EMA Smoothing & Deadband ──
        current = self.positions[joint]
        diff = target_pos - current
        
        if abs(diff) > DEADBAND:
            # Smooth the movement: new_pos = current + alpha * diff
            new_pos = int(current + (SMOOTH_FACTOR * diff))
            self.positions[joint] = new_pos
            self._write(SERVO_IDS[joint], new_pos)

    def close(self):
        if self.connected:
            self.disable_torque()
            self.port.closePort()
            print("[OK]  Port closed.")

# ── Gesture recognition ───────────────────────────────────────────────────────
def count_fingers(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = []
    if handedness_label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    dx, dy = lm[9].x - lm[0].x, lm[9].y - lm[0].y
    tilt = math.degrees(math.atan2(-dy, dx))
    return fingers, tilt

def classify_gesture(fingers, tilt, landmarks=None):
    t, i, m, r, p = fingers
    ext = sum(fingers)

    if landmarks:
        d = math.sqrt((landmarks[4].x - landmarks[8].x) ** 2 +
                      (landmarks[4].y - landmarks[8].y) ** 2)
        if d < 0.04 and m == 1 and r == 1 and p == 1:
            return "OK_SIGN"

    if ext == 0: return "FIST"
    if ext == 1 and i == 1: return "INDEX_UP"
    if ext == 1 and t == 1: return "THUMB"
    if ext == 2 and i == 1 and m == 1: return "PEACE"
    if ext == 2 and i == 1 and t == 1: return "L_SHAPE"
    if ext == 2 and i == 1 and p == 1: return "HORNS"
    if ext == 3 and i == 1 and m == 1 and r == 1: return "THREE"
    if ext == 4 and i == 1 and m == 1 and r == 1 and p == 1: return "FOUR"
    if ext == 5: return "OPEN_HAND"
    return "UNKNOWN"

GESTURE_LABELS = {
    "IMITATION":  "🤖 Total Imitation Mode",
    "FIST":       "✊ Gripper Closed",
    "OPEN_HAND":  "🖐 Gripper Open",
    "UNKNOWN":    "Searching for hand...",
}

# ── Command application ───────────────────────────────────────────────────────
def apply_commands(gesture, fingers, landmarks, robot: ServoController,
                   hand_center=None, wrist_depth_m: float = None):
    """
    TOTAL IMITATION MAPPING:
      Hand X/Y → Shoulder Pan/Tilt
      Depth Z  → Elbow
      Hand Roll → Wrist Rotation
      Hand Pitch → Wrist Flex
      Fingers Extended → Gripper
    """
    if not landmarks:
        return

    # 1. Shoulder Pan/Tilt (Positional)
    if hand_center:
        hx, hy = hand_center
        # Pan: mirror hand X
        lo_p, hi_p = LIMITS["shoulder_pan"]
        target_pan = int(lo_p + (hi_p - lo_p) * (1.0 - hx))
        robot.set_pos("shoulder_pan", target_pan)

        # Tilt: map hand Y
        lo_t, hi_t = LIMITS["shoulder_tilt"]
        target_tilt = int(lo_t + (hi_t - lo_t) * hy)
        robot.set_pos("shoulder_tilt", target_tilt)

    # 2. Elbow (Depth-driven)
    if wrist_depth_m is not None and DEPTH_AVAILABLE:
        target_elbow = depth_to_elbow_pos(wrist_depth_m)
        robot.set_pos("elbow", target_elbow)

    # 3. Wrist Rotation (Roll)
    # Calculate angle between Wrist (0) and Middle Finger Base (9)
    p0 = landmarks[0]
    p9 = landmarks[9]
    roll_rad = math.atan2(p9.y - p0.y, p9.x - p0.x)
    roll_deg = math.degrees(roll_rad) + 90 # Offset to center it
    
    lo_r, hi_r = LIMITS["wrist_rot"]
    # Map -60 to 60 degrees to servo range
    roll_pct = np.clip((roll_deg + 60) / 120, 0, 1)
    target_rot = int(lo_r + (hi_r - lo_r) * roll_pct)
    robot.set_pos("wrist_rot", target_rot)

    # 4. Wrist Flex (Pitch)
    # Estimate pitch by looking at the distance between wrist and middle finger tip
    # A "flat" hand is long; a "tilted" hand looks shorter to the camera
    dist = math.sqrt((p9.x - p0.x)**2 + (p9.y - p0.y)**2)
    # Normalize by hand size (distance between wrist and index mcp)
    p5 = landmarks[5]
    ref_dist = math.sqrt((p5.x - p0.x)**2 + (p5.y - p0.y)**2)
    pitch_factor = np.clip(dist / (ref_dist * 1.5), 0.5, 1.2)
    
    lo_f, hi_f = LIMITS["wrist_flex"]
    target_flex = int(lo_f + (hi_f - lo_f) * (pitch_factor - 0.5) * 2)
    robot.set_pos("wrist_flex", target_flex)

    # 5. Gripper (Openness)
    # Map finger count (0-5) to Gripper (Closed-Open)
    ext_count = sum(fingers)
    lo_g, hi_g = LIMITS["gripper"]
    target_grip = int(lo_g + (hi_g - lo_g) * (ext_count / 5.0))
    robot.set_pos("gripper", target_grip)

# ── HUD overlay ───────────────────────────────────────────────────────────────
COLORS = {
    "FIST": (60, 60, 220), "INDEX_UP": (50, 200, 50), "L_SHAPE": (50, 200, 200),
    "THUMB": (200, 130, 0), "PEACE": (0, 180, 200), "HORNS": (180, 0, 200),
    "THREE": (0, 120, 255), "FOUR": (0, 80, 255), "OPEN_HAND": (220, 200, 0),
    "OK_SIGN": (0, 220, 180), "UNKNOWN": (120, 120, 120),
}

def draw_overlay(frame, gesture, tilt, robot: ServoController, fps,
                 wrist_depth_m=None, wrist_px=None):
    h, w = frame.shape[:2]
    color = COLORS.get(gesture, (120, 120, 120))

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SO-101  |  Depth Gesture Control",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, GESTURE_LABELS.get(gesture, "?"),
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Depth readout
    if wrist_depth_m is not None:
        depth_txt = f"Depth: {wrist_depth_m:.3f} m"
        depth_col = (0, 230, 200) if DEPTH_AVAILABLE else (120, 120, 120)
        cv2.putText(frame, depth_txt, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, depth_col, 2)

        # Visual depth bar
        bar_pct = np.clip((wrist_depth_m - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0, 1)
        cv2.rectangle(frame, (10, 98), (280, 110), (50, 50, 50), 1)
        cv2.rectangle(frame, (10, 98), (10 + int(270 * bar_pct), 110), depth_col, -1)

    # Wrist dot on frame
    if wrist_px:
        cx, cy = wrist_px
        cv2.circle(frame, (cx, cy), 10, color, -1)
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)

    # Joint position bars
    y = 128
    for joint, pos in robot.positions.items():
        lo, hi = LIMITS[joint]
        pct = (pos - lo) / (hi - lo)
        cv2.putText(frame, f"{joint[:12]}: {pos}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.rectangle(frame, (140, y - 10), (280, y), (50, 50, 50), 1)
        cv2.rectangle(frame, (140, y - 10), (140 + int(pct * 140), y), color, -1)
        y += 22

    # FPS & connection status
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    status_color = (0, 255, 0) if robot.connected else (0, 165, 255)
    cv2.circle(frame, (w - 20, 20), 8, status_color, -1)
    cv2.putText(frame, "LIVE" if robot.connected else "SIM",
                (w - 65, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

# ── Port listing helper ────────────────────────────────────────────────────────
def list_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found.")
        return
    print("Available COM Ports:")
    for p in ports:
        print(f"  - {p.device}: {p.description}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SO-101 Depth + Gesture Robot Arm Control")
    parser.add_argument("--port",    type=str,  default=DEFAULT_PORT,
                        help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud",    type=int,  default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--sim",     action="store_true", help="Run in simulation mode")
    parser.add_argument("--list",    action="store_true", help="List COM ports and exit")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable Depth Anything (gesture-only mode)")
    parser.add_argument("--depth-model", type=str, default=DEPTH_MODEL_PATH,
                        help=f"Path to .pth weights (default: {DEPTH_MODEL_PATH})")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    # ── Load depth model ──────────────────────────────────────────────────
    use_depth = DEPTH_AVAILABLE and not args.no_depth
    depth_model = None
    if use_depth:
        try:
            depth_model = load_depth_model(args.depth_model)
        except Exception as e:
            print(f"[WARN] Could not load depth model: {e}. Depth disabled.")
            use_depth = False

    # ── Init robot & camera ───────────────────────────────────────────────
    robot = ServoController(args.port, args.baud, args.sim)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam.")
        robot.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    interval   = 1.0 / FPS_TARGET
    last_time  = 0.0
    fps        = 0.0
    gesture    = "UNKNOWN"
    tilt       = 0.0
    wrist_depth_m  = None
    wrist_px       = None

    print("\n[INFO] Controls: 'q' quit | 'r' reset arm\n")

    try:
        with mp_hands.Hands(model_complexity=0,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.5) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                now = time.time()

                if now - last_time >= interval:
                    elapsed   = max(now - last_time, 1e-6)
                    fps       = 1.0 / elapsed
                    last_time = now

                    # ── Depth inference ───────────────────────────────────
                    depth_map = None
                    if use_depth and depth_model is not None:
                        with torch.no_grad():
                            depth_map = depth_model.infer_image(frame)   # H×W float32
                            # Normalize to 0.0 - 1.0 for relative models
                            d_min, d_max = depth_map.min(), depth_map.max()
                            if d_max > d_min:
                                depth_map = (depth_map - d_min) / (d_max - d_min)

                    # ── Hand tracking ─────────────────────────────────────
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        hl   = results.multi_hand_landmarks[0]
                        side = results.multi_handedness[0].classification[0].label
                        mp_drawing.draw_landmarks(
                            frame, hl, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style())

                        fingers, tilt = count_fingers(hl, side)
                        gesture       = classify_gesture(fingers, tilt, hl.landmark)

                        # Wrist pixel coords
                        h_f, w_f, _ = frame.shape
                        wrist = hl.landmark[mp_hands.HandLandmark.WRIST]
                        cx = int(np.clip(wrist.x * w_f, 0, w_f - 1))
                        cy = int(np.clip(wrist.y * h_f, 0, h_f - 1))
                        wrist_px = (cx, cy)

                        # Sample metric depth at wrist
                        if depth_map is not None:
                            wrist_depth_m = get_depth_at(depth_map, cx, cy)
                        else:
                            wrist_depth_m = None

                        apply_commands(
                            gesture, fingers, hl.landmark, robot,
                            hand_center=(wrist.x, wrist.y),
                            wrist_depth_m=wrist_depth_m,
                        )
                    else:
                        gesture       = "FIST"
                        wrist_depth_m = None
                        wrist_px      = None

                draw_overlay(frame, gesture, tilt, robot, fps,
                             wrist_depth_m=wrist_depth_m, wrist_px=wrist_px)
                cv2.imshow("SO-101  |  Depth Gesture Control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    for j in SERVO_IDS:
                        robot.set_pos(j, 2048)
                    print("[OK] Arm reset to home position.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close()
        print("[OK] Finished.")


if __name__ == "__main__":
    main()