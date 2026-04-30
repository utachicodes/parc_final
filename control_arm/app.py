"""
SO-101 Robot Controller - Flask version
Run: python app.py  →  http://localhost:5000
"""
import sys, os, time, threading, queue, json, subprocess, signal, select
from pathlib import Path
SDK_PATH = Path(__file__).parent / "stservo-env"
if not SDK_PATH.exists():
    SDK_PATH = Path(__file__).parent / "STServo_Python"
sys.path.insert(0, str(SDK_PATH))

import numpy as np
import cv2
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import serial.tools.list_ports
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS

# ── logging (SSE broadcast) ───────────────────────────────────────────────────
_log_entries: list[str] = []
_log_lock = threading.Lock()
_log_subs: list[queue.Queue] = []

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    with _log_lock:
        _log_entries.append(entry)
        if len(_log_entries) > 300:
            _log_entries.pop(0)
        for q in _log_subs:
            try: q.put_nowait(entry)
            except queue.Full: pass

# ── i18n ────────────────────────────────────────────────────────────────────────
_translations: dict = {}
_current_lang = "en"

def load_translations():
    global _translations, _current_lang
    trans_dir = Path(__file__).parent / "translations"
    _translations = {"en": {}, "fr": {}}
    for lang in ["en", "fr"]:
        f = trans_dir / f"{lang}.json"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fp:
                _translations[lang] = json.load(fp)
    log(f"Loaded translations: {list(_translations.keys())}")

def t(key: str, lang: str = None) -> str:
    """Translate key, supports dot notation like 'cards.play.title'"""
    lang = lang or _current_lang
    keys = key.split(".")
    val = _translations.get(lang, {})
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, key)
        else:
            return key
    return val if isinstance(val, str) else key

load_translations()

# ── servo IDs ─────────────────────────────────────────────────────────────────
DEGREE_TO_UNIT = 4095 / 360
SERVO_IDS: dict = {}
SERVO_INVERT: dict = {}  # Direction flip: 1 = normal, -1 = inverted

def load_servo_ids():
    global SERVO_IDS, SERVO_INVERT
    ids_file = Path(__file__).parent / "servo_ids.json"
    defaults = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
                "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
    invert_file = Path(__file__).parent / "servo_invert.json"
    if ids_file.exists():
        try:
            with open(ids_file) as f:
                SERVO_IDS = json.load(f)
        except:
            SERVO_IDS = defaults
    else:
        SERVO_IDS = defaults
    if invert_file.exists():
        try:
            with open(invert_file) as f:
                SERVO_INVERT = json.load(f)
        except:
            SERVO_INVERT = {}
    if not SERVO_INVERT:
        SERVO_INVERT = {k: 1 for k in SERVO_IDS}

load_servo_ids()

# ── CalibrationData ───────────────────────────────────────────────────────────
class CalibrationData:
    def __init__(self):
        self.min_pos: dict = {}
        self.max_pos: dict = {}
        self.center_pos: dict = {}
        self.home_pos: dict = {}
        self.min_deg: dict = {}
        self.max_deg: dict = {}
        self.center_deg: dict = {}
        self.calibrated = False
        self.sweep_speed = 800
        self.sweep_acc   = 50
        self.comm_errors = 0
        self.dropped_reads = 0
        self._stop_event = threading.Event()

calibration = CalibrationData()

# ── Robot ─────────────────────────────────────────────────────────────────────
class Robot:
    def __init__(self, port: str, baudrate: int = 1_000_000):
        self.port_name = port
        self.baudrate  = baudrate
        self.port_handler = None
        self.servo = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self) -> tuple[bool, str]:
        try:
            self.port_handler = PortHandler(self.port_name)
            if not self.port_handler.openPort():
                return False, f"Failed to open {self.port_name}"
            if not self.port_handler.setBaudRate(self.baudrate):
                self.port_handler.closePort()
                return False, f"Failed to set baudrate {self.baudrate}"
            self.servo = sms_sts(self.port_handler)
            self.connected = True
            log(f"Connected to {self.port_name}")
            return True, "Connected"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        if self.port_handler:
            self.port_handler.closePort()
        self.connected = False

    def send_positions(self, positions: dict, speed: int = 500, acc: int = 50):
        if not self.connected or not self.servo: return False
        try:
            with self._lock:
                for name, pos_deg in positions.items():
                    if name in SERVO_IDS:
                        inv = SERVO_INVERT.get(name, 1)
                        pos_unit = max(0, min(4095, int(2048 + pos_deg * inv * DEGREE_TO_UNIT)))
                        self.servo.WritePosEx(SERVO_IDS[name], pos_unit, speed, acc)
            return True
        except Exception as e:
            log(f"send_positions error: {e}")
            return False

    def read_raw_positions(self) -> dict:
        positions = {}
        if self.connected and self.servo:
            with self._lock:
                for name, sid in SERVO_IDS.items():
                    pos, result, _ = self.servo.ReadPos(sid)
                    if result == COMM_SUCCESS and isinstance(pos, (int, float)):
                        positions[name] = int(pos)
                    else:
                        positions[name] = 2048
        return positions

    def get_joint_states(self) -> dict:
        """
        Read joint positions in degrees and compute FK position.
        Uses Tucker course formulas from ECE 4560 Assignment 6.

        Returns:
            dict with:
                - joints: {joint_name: angle_in_degrees}
                - raw: {joint_name: raw_servo_value}
                - fk_position: [x, y, z] in meters
                - fk_rotation: 3x3 rotation matrix
        """
        raw = self.read_raw_positions()
        joints = {}
        for name, val in raw.items():
            inv = SERVO_INVERT.get(name, 1)
            deg = (val - 2048) * inv / DEGREE_TO_UNIT
            joints[name] = round(deg, 2)

        fk_pos, fk_rot = self.compute_fk(joints)

        return {
            'joints': joints,
            'raw': raw,
            'fk_position': fk_pos.tolist() if hasattr(fk_pos, 'tolist') else fk_pos,
            'fk_rotation': fk_rot.tolist() if hasattr(fk_rot, 'tolist') else fk_rot
        }

    @staticmethod
    def compute_fk(joints: dict) -> tuple:
        """
        Compute forward kinematics using Tucker course formulas.

        IMPORTANT:
        - Input: joint angles in DEGREES
        - Output: position in METERS, rotation as 3x3 matrix

        Based on: https://maegantucker.com/ECE4560/assignment6-so101/
        """
        def Rx(thetadeg):
            th = np.deg2rad(thetadeg)
            c, s = np.cos(th), np.sin(th)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def Ry(thetadeg):
            th = np.deg2rad(thetadeg)
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def Rz(thetadeg):
            th = np.deg2rad(thetadeg)
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        t1 = joints.get('shoulder_pan', 0)
        t2 = joints.get('shoulder_lift', 0)
        t3 = joints.get('elbow_flex', 0)
        t4 = joints.get('wrist_flex', 0)
        t5 = joints.get('wrist_roll', 0)

        gw1 = np.block([[Rz(180) @ Rx(180) @ Rz(t1),
                         np.array([0.0388353, 0.0, 0.0624]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g12 = np.block([[Rx(-90 - t2), np.array([0, 0, 0.100]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g23 = np.block([[Rx(90 + t3), np.array([0.100, 0, 0]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g34 = np.block([[Rx(-90 - t4), np.array([0, 0, 0.080]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g45 = np.block([[Rz(t5), np.array([0, 0, 0.050]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g5t = np.block([[np.eye(3), np.array([0, 0, 0.050]).reshape(3, 1)],
                        [0, 0, 0, 1]])

        g0t = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t

        position = g0t[0:3, 3]
        rotation = g0t[0:3, 0:3]

        return position, rotation

    @staticmethod
    def compute_ik(target_position: list, target_rotation: list = None, initial_joints: dict = None, max_iterations: int = 300, tolerance: float = 0.005) -> dict:
        """
        Compute Inverse Kinematics using Jacobian Transpose method.

        Args:
            target_position: [x, y, z] in meters
            target_rotation: Optional 3x3 rotation matrix (uses default if None)
            initial_joints: Starting joint configuration (uses home if None)
            max_iterations: Maximum IK iterations
            tolerance: Position error tolerance in meters

        Returns:
            dict with 'joints' (solution), 'error' (final error), 'success' (bool)
        """
        # Link lengths
        l1 = 0.0624  # base to j1
        l2 = 0.100   # j1 to j2
        l3 = 0.100   # j2 to j3
        l4 = 0.080   # j3 to j4
        l5 = 0.050   # j4 to j5
        l6 = 0.050   # j5 to tip

        # Total reach
        max_reach = l2 + l3 + l4 + l5 + l6

        # Check if target is reachable
        target = np.array(target_position)
        distance = np.linalg.norm(target)
        if distance > max_reach * 0.95:
            return {'joints': None, 'error': 'Target out of reach', 'success': False}

        # Initialize joints (home position or provided)
        # API expects degrees, but solver operates in radians internally
        if initial_joints:
            joints = np.array([
                np.deg2rad(initial_joints.get('shoulder_pan', 0)),
                np.deg2rad(initial_joints.get('shoulder_lift', 0)),
                np.deg2rad(initial_joints.get('elbow_flex', 0)),
                np.deg2rad(initial_joints.get('wrist_flex', 0)),
                np.deg2rad(initial_joints.get('wrist_roll', 0))
            ])
        else:
            joints = np.zeros(5)

        alpha = 0.5  # Learning rate
        damping = 0.1  # Damping factor for numerical stability

        for iteration in range(max_iterations):
            # Current end effector position (convert radians to degrees for FK)
            current_joints = {
                'shoulder_pan': np.rad2deg(joints[0]),
                'shoulder_lift': np.rad2deg(joints[1]),
                'elbow_flex': np.rad2deg(joints[2]),
                'wrist_flex': np.rad2deg(joints[3]),
                'wrist_roll': np.rad2deg(joints[4])
            }
            pos, _ = Robot.compute_fk(current_joints)

            # Error
            error = target - pos
            if np.linalg.norm(error) < tolerance:
                return {
                    'joints': {
                        'shoulder_pan': round(float(np.rad2deg(joints[0])), 2),
                        'shoulder_lift': round(float(np.rad2deg(joints[1])), 2),
                        'elbow_flex': round(float(np.rad2deg(joints[2])), 2),
                        'wrist_flex': round(float(np.rad2deg(joints[3])), 2),
                        'wrist_roll': round(float(np.rad2deg(joints[4])), 2)
                    },
                    'error': np.linalg.norm(error),
                    'success': True,
                    'iterations': iteration
                }

            # Numerical Jacobian (5 columns for 5 joints)
            # Note: delta in radians, but FK expects degrees, so convert
            delta_rad = 0.01
            jacobian = np.zeros((3, 5))
            for i in range(5):
                joints_plus = joints.copy()
                joints_plus[i] += delta_rad
                test_joints = {
                    'shoulder_pan': np.rad2deg(joints_plus[0]),
                    'shoulder_lift': np.rad2deg(joints_plus[1]),
                    'elbow_flex': np.rad2deg(joints_plus[2]),
                    'wrist_flex': np.rad2deg(joints_plus[3]),
                    'wrist_roll': np.rad2deg(joints_plus[4])
                }
                pos_plus, _ = Robot.compute_fk(test_joints)
                jacobian[:, i] = (pos_plus - pos) / delta_rad

            # Damped least squares (Levenberg-Marquardt)
            # delta_theta = (J^T * J + lambda^2 * I)^(-1) * J^T * error
            jacobian_T = jacobian.T
            damping_matrix = damping * damping * np.eye(5)
            delta_joints = np.linalg.inv(jacobian_T @ jacobian + damping_matrix) @ jacobian_T @ error

            joints = joints + alpha * delta_joints

            # Apply joint limits (convert degree limits to radians)
            limits = [
                (-np.deg2rad(110), np.deg2rad(110)),   # shoulder_pan
                (-np.deg2rad(100), np.deg2rad(90)),    # shoulder_lift
                (-np.deg2rad(140), np.deg2rad(140)),   # elbow_flex
                (-np.deg2rad(100), np.deg2rad(100)),   # wrist_flex
                (-np.deg2rad(150), np.deg2rad(150))    # wrist_roll
            ]
            for i, (min_lim, max_lim) in enumerate(limits):
                joints[i] = np.clip(joints[i], min_lim, max_lim)

        return {
            'joints': {
                'shoulder_pan': round(float(np.rad2deg(joints[0])), 2),
                'shoulder_lift': round(float(np.rad2deg(joints[1])), 2),
                'elbow_flex': round(float(np.rad2deg(joints[2])), 2),
                'wrist_flex': round(float(np.rad2deg(joints[3])), 2),
                'wrist_roll': round(float(np.rad2deg(joints[4])), 2)
            },
            'error': float(np.linalg.norm(error)),
            'success': False,
            'iterations': max_iterations
        }

    def write_raw_position(self, servo_id: int, pos: int, speed: int = 200, acc: int = 20, invert: int = 1):
        if not self.connected or not self.servo:
            log(f"write_raw ID{servo_id}: robot non connecté")
            return False
        try:
            with self._lock:
                pos_inv = int(2048 + (pos - 2048) * invert)
                self.servo.WritePosEx(servo_id, max(0, min(4095, pos_inv)), speed, acc)
            return True
        except Exception as e:
            log(f"write_raw error ID{servo_id}: {e}")
            return False

    def auto_calibrate(self, limits_deg: dict | None = None) -> tuple[bool, str]:
        global calibration
        calibration._stop_event.clear()
        try:
            log("Reading servo hardware limits from EEPROM...")
        
            # Read hardware limits from each servo first
            hw_limits = {}
            for name, sid in SERVO_IDS.items():
                if calibration._stop_event.is_set():
                    return False, "Stopped"
                try:
                    with self._lock:
                        raw_cw_limit, res_cw, _ = self.servo.read2ByteTxRx(sid, 6)
                        raw_ccw_limit, res_ccw, _ = self.servo.read2ByteTxRx(sid, 8)
                        raw_max, res_max, _ = self.servo.read2ByteTxRx(sid, 9)
                        raw_min, res_min, _ = self.servo.read2ByteTxRx(sid, 11)
                    log(f"  {name} EEPROM: addr6(CW)={raw_cw_limit}(res={res_cw}), addr8(CCW)={raw_ccw_limit}(res={res_ccw}), addr9={raw_max}(res={res_max}), addr11={raw_min}(res={res_min})")
                    if res_max == COMM_SUCCESS and res_min == COMM_SUCCESS and raw_min < raw_max:
                        hw_limits[name] = {
                            'min': raw_min,
                            'max': raw_max,
                            'raw_min': raw_min,
                            'raw_max': raw_max
                        }
                        log(f"  {name}: HW limits {raw_min}-{raw_max}")
                    elif res_cw == COMM_SUCCESS and res_ccw == COMM_SUCCESS and raw_cw_limit < raw_ccw_limit:
                        hw_limits[name] = {
                            'min': raw_ccw_limit,
                            'max': raw_cw_limit,
                            'raw_min': raw_ccw_limit,
                            'raw_max': raw_cw_limit
                        }
                        log(f"  {name}: Angle limits {raw_ccw_limit}-{raw_cw_limit}")
                except Exception as e:
                    log(f"  Failed to read HW limits for {name}: {e}")
            
            log("Starting auto-calibration...")
            current = self.read_raw_positions()
        
            for name, pos in current.items():
                if limits_deg and name in limits_deg:
                    lim = limits_deg[name]
                    calibration.center_pos[name] = max(0, min(4095, int(2048 + lim['ctr'] * DEGREE_TO_UNIT)))
                    calibration.min_pos[name]    = max(0, min(4095, int(2048 + lim['min'] * DEGREE_TO_UNIT)))
                    calibration.max_pos[name]    = max(0, min(4095, int(2048 + lim['max'] * DEGREE_TO_UNIT)))
                elif name in hw_limits:
                    # Use hardware limits with margin
                    calibration.min_pos[name]    = hw_limits[name]['min']
                    calibration.max_pos[name]    = hw_limits[name]['max']
                    calibration.center_pos[name] = pos  # Start from current position
                else:
                    calibration.center_pos[name] = pos
                    calibration.min_pos[name]    = 0
                    calibration.max_pos[name]    = 4095

            for name, sid in SERVO_IDS.items():
                if calibration._stop_event.is_set():
                    log("⏹ Calibration stopped by the user")
                    return False, "Stopped"
                log(f"Sweeping {name} (ID:{sid})...")
                inv = SERVO_INVERT.get(name, 1)
                center   = calibration.center_pos[name]
                safe_min = calibration.min_pos[name]
                safe_max = calibration.max_pos[name]
                if inv < 0:
                    safe_min, safe_max = safe_max, safe_min
                self.write_raw_position(sid, int(safe_min), speed=calibration.sweep_speed, acc=calibration.sweep_acc)
                time.sleep(2.5)
                if calibration._stop_event.is_set():
                    self.write_raw_position(sid, int(center), speed=400, acc=40)
                    log("⏹ Calibration stopped by the user")
                    return False, "Stopped"
                time.sleep(0.1)
                try:
                    if self.servo:
                        r = self.servo.ReadPos(sid)
                        log(f"  {name} min read: pos={r[0] if r else 'None'}, result={r[1] if r else 'None'}")
                        if r and r[1] == COMM_SUCCESS and r[0] is not None:
                            calibration.min_pos[name] = int(r[0])
                except: calibration.comm_errors += 1
                self.write_raw_position(sid, int(center), speed=calibration.sweep_speed, acc=calibration.sweep_acc)
                time.sleep(1.5)
                self.write_raw_position(sid, int(safe_max), speed=calibration.sweep_speed, acc=calibration.sweep_acc)
                time.sleep(2.5)
                if calibration._stop_event.is_set():
                    self.write_raw_position(sid, int(center), speed=400, acc=40)
                    log("⏹ Calibration stoppée par l'utilisateur")
                    return False, "Stopped"
                time.sleep(0.1)
                try:
                    if self.servo:
                        r = self.servo.ReadPos(sid)
                        log(f"  {name} max read: pos={r[0] if r else 'None'}, result={r[1] if r else 'None'}")
                        if r and r[1] == COMM_SUCCESS and r[0] is not None:
                            calibration.max_pos[name] = int(r[0])
                except: calibration.comm_errors += 1
                self.write_raw_position(sid, int(center), speed=calibration.sweep_speed, acc=calibration.sweep_acc)
                time.sleep(1)
                calibration.center_pos[name] = (calibration.min_pos[name] + calibration.max_pos[name]) // 2
            calibration.calibrated = True
            for name in SERVO_IDS:
                inv = SERVO_INVERT.get(name, 1)
                if inv < 0:
                    calibration.min_pos[name], calibration.max_pos[name] = calibration.max_pos[name], calibration.min_pos[name]
                    calibration.center_pos[name] = (calibration.min_pos[name] + calibration.max_pos[name]) // 2
                calibration.center_deg[name] = round((calibration.center_pos.get(name, 2048) - 2048) / DEGREE_TO_UNIT, 1)
                calibration.min_deg[name]    = round((calibration.min_pos.get(name, 1548)   - 2048) / DEGREE_TO_UNIT, 1)
                calibration.max_deg[name]    = round((calibration.max_pos.get(name, 2548)   - 2048) / DEGREE_TO_UNIT, 1)
            cal_file = Path(__file__).parent / 'calibration.json'
            save_data = {
                'limits_deg': {k: {'min': calibration.min_deg.get(k, -90),
                                   'ctr': calibration.center_deg.get(k, 0),
                                   'max': calibration.max_deg.get(k, 90)}
                              for k in SERVO_IDS},
                'min_pos':    calibration.min_pos,
                'max_pos':    calibration.max_pos,
                'center_pos': calibration.center_pos,
            }
            with open(cal_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            log(f"✔ Auto-calibration complete!")
            log(f"  Saved to {cal_file.name}")
            return True, "Calibration successful"
        except Exception as e:
            log(f"Auto-calibration error: {e}")
            return False, str(e)

class DummyRobot:
    connected = False
    servo = None
    def connect(self): return False, "No hardware"
    def disconnect(self): pass
    def send_positions(self, positions, speed=500, acc=50): pass
    def read_raw_positions(self): return {}
    def write_raw_position(self, sid, pos, speed=200, acc=20): pass

# ── FaceDetector ──────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        self.backend = "none"
        self._mp_face_mesh = None
        self._mp_hand_detector = None
        self._mp_pose_detector = None
        self._cv2_face_cascade = None
        self._ts = 0
        self._setup()

    def _setup(self):
        import os, urllib.request
        mediapipe_available = False
        self._use_gpu = False

        try:
            from mediapipe.tasks.python.vision import (FaceLandmarker, HandLandmarker,
                FaceLandmarkerOptions, HandLandmarkerOptions)
            from mediapipe.tasks.python.vision import (PoseLandmarker, PoseLandmarkerOptions)
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision import RunningMode
            import mediapipe as mp
            self._mp_image_class  = mp.Image
            self._mp_image_format = mp.ImageFormat
            self._mp_drawing = mp.solutions.drawing_utils
            mediapipe_available = True
            log("MediaPipe available, initializing detectors with GPU...")
        except ImportError:
            log("MediaPipe not available, using OpenCV fallback")

        # Initialize OpenCV cascade as fallback (works without mediapipe)
        try:
            cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._cv2_face_cascade = cv2.CascadeClassifier(cascade)
            if not self._cv2_face_cascade.empty():
                self.backend = "opencv_cascade"
                log("Face detector: OpenCV Haar Cascade")
        except Exception as e2:
            log(f"OpenCV cascade failed: {e2}")

        if not mediapipe_available:
            return

        # Check for GPU availability and enable GPU delegate
        try:
            import torch
            if torch.cuda.is_available():
                self._use_gpu = True
                log("GPU detected - MediaPipe will use GPU acceleration")
        except ImportError:
            pass

        HAND_URL = ("https://storage.googleapis.com/mediapipe-assets/"
                    "hand_landmarker.task")
        try:
            hand_model = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
            if os.path.exists(hand_model):
                try:
                    base_opts = base_options.BaseOptions(model_asset_path=hand_model)
                    opts = HandLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=RunningMode.VIDEO, num_hands=1)
                    self._mp_hand_detector = HandLandmarker.create_from_options(opts)
                except:
                    if os.path.exists(hand_model):
                        os.remove(hand_model)
            if not os.path.exists(hand_model):
                log("Downloading hand model...")
                urllib.request.urlretrieve(HAND_URL, hand_model)
                base_opts = base_options.BaseOptions(model_asset_path=hand_model)
                opts = HandLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=RunningMode.VIDEO, num_hands=1)
                self._mp_hand_detector = HandLandmarker.create_from_options(opts)
            log("Hand detector: MediaPipe Tasks" + (" (GPU)" if self._use_gpu else ""))
        except Exception as e:
            log(f"Hand detector failed: {e}")

        FACEMESH_URL = ("https://storage.googleapis.com/mediapipe-assets/"
                        "face_landmarker.task")
        try:
            face_model = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if os.path.exists(face_model):
                try:
                    base_opts = base_options.BaseOptions(model_asset_path=face_model)
                    opts = FaceLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=RunningMode.VIDEO, num_faces=1)
                    self._mp_face_mesh = FaceLandmarker.create_from_options(opts)
                except:
                    log("Cached face mesh model invalid, re-downloading...")
                    if os.path.exists(face_model):
                        os.remove(face_model)
            if not os.path.exists(face_model):
                log("Downloading face mesh model...")
                urllib.request.urlretrieve(FACEMESH_URL, face_model)
            base_opts = base_options.BaseOptions(model_asset_path=face_model)
            opts = FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=RunningMode.VIDEO, num_faces=1)
            self._mp_face_mesh = FaceLandmarker.create_from_options(opts)
            self.backend = "mediapipe_face_mesh"
            log("Face detector: MediaPipe Face Mesh (478 landmarks)" + (" (GPU)" if self._use_gpu else ""))
        except Exception as e:
            log(f"MediaPipe face mesh failed: {e}")

        POSE_URL = ("https://storage.googleapis.com/mediapipe-assets/"
                    "pose_landmarker.task")
        try:
            pose_model = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
            if os.path.exists(pose_model):
                try:
                    base_opts = base_options.BaseOptions(model_asset_path=pose_model)
                    opts = PoseLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=RunningMode.VIDEO, num_poses=1)
                    self._mp_pose_detector = PoseLandmarker.create_from_options(opts)
                except:
                    log("Cached pose model invalid, re-downloading...")
                    if os.path.exists(pose_model):
                        os.remove(pose_model)
            if not os.path.exists(pose_model):
                log("Downloading pose model...")
                urllib.request.urlretrieve(POSE_URL, pose_model)
                base_opts = base_options.BaseOptions(model_asset_path=pose_model)
                opts = PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=RunningMode.VIDEO, num_poses=1)
                self._mp_pose_detector = PoseLandmarker.create_from_options(opts)
            log("Pose detector: MediaPipe Pose (33 landmarks)" + (" (GPU)" if self._use_gpu else ""))
        except Exception as e:
            log(f"MediaPipe pose failed: {e}")

    def _next_ts(self) -> int:
        self._ts += 1
        return self._ts

    def detect_face(self, bgr) -> tuple | None:
        if self.backend == "mediapipe_face_mesh" and self._mp_face_mesh:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = self._mp_image_class(image_format=self._mp_image_format.SRGB, data=rgb)
                r = self._mp_face_mesh.detect_for_video(mp_img, self._next_ts())
                if r.face_landmarks and len(r.face_landmarks) > 0:
                    landmarks = r.face_landmarks[0]
                    nose_tip = landmarks[1]
                    cx = nose_tip.x
                    cy = nose_tip.y
                    return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), landmarks)
            except: pass
        if self.backend == "opencv_cascade" and self._cv2_face_cascade:
            small = cv2.resize(bgr, (160, 120))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self._cv2_face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(20, 20))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return (float((x + w/2) / 160), float((y + h/2) / 120), None)
        return None

    def detect_hand(self, bgr) -> tuple | None:
        if self._mp_hand_detector:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = self._mp_image_class(image_format=self._mp_image_format.SRGB, data=rgb)
                r = self._mp_hand_detector.detect_for_video(mp_img, self._next_ts())
                if r.hand_landmarks:
                    lm = r.hand_landmarks[0]
                    cx = (lm[9].x + lm[13].x) / 2
                    cy = (lm[9].y + lm[13].y) / 2
                    side = "Right"
                    if r.handedness:
                        side = r.handedness[0][0].category_name
                    return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), side, r.hand_landmarks[0])
            except: pass
        return None

    def draw_hand_mesh(self, frame, landmarks, handedness="Right"):
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]
        color = (255, 150, 80) if handedness == "Right" else (80, 200, 255)
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 2, color, -1)
        finger_tips = [4, 8, 12, 16, 20]
        for idx in finger_tips:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        return frame

    def detect_pose(self, bgr) -> tuple | None:
        if self._mp_pose_detector:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = self._mp_image_class(image_format=self._mp_image_format.SRGB, data=rgb)
                r = self._mp_pose_detector.detect_for_video(mp_img, self._next_ts())
                if r.pose_landmarks and len(r.pose_landmarks) > 0:
                    landmarks = r.pose_landmarks[0]
                    nose = landmarks[0]
                    cx = nose.x
                    cy = nose.y
                    return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), r.pose_landmarks[0])
            except: pass
        return None

    def detect_pose_full(self, bgr) -> dict | None:
        if self._mp_pose_detector:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = self._mp_image_class(image_format=self._mp_image_format.SRGB, data=rgb)
                r = self._mp_pose_detector.detect_for_video(mp_img, self._next_ts())
                if r.pose_landmarks and len(r.pose_landmarks) > 0:
                    lm = r.pose_landmarks[0]
                    return {
                        'nose': lm[0],
                        'left_shoulder': lm[11], 'right_shoulder': lm[12],
                        'left_elbow': lm[13], 'right_elbow': lm[14],
                        'left_wrist': lm[15], 'right_wrist': lm[16],
                        'left_hip': lm[23], 'right_hip': lm[24],
                        'landmarks': lm
                    }
            except: pass
        return None

    def detect_pose_detail(self, bgr) -> dict | None:
        if self._mp_pose_detector:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = self._mp_image_class(image_format=self._mp_image_format.SRGB, data=rgb)
                r = self._mp_pose_detector.detect_for_video(mp_img, self._next_ts())
                if r.pose_landmarks and len(r.pose_landmarks) > 0:
                    lm = r.pose_landmarks[0]
                    return {
                        'landmarks': lm,
                        'joints': {
                            0: ('nose', lm[0]),
                            11: ('L_shoulder', lm[11]), 12: ('R_shoulder', lm[12]),
                            13: ('L_elbow', lm[13]), 14: ('R_elbow', lm[14]),
                            15: ('L_wrist', lm[15]), 16: ('R_wrist', lm[16]),
                            23: ('L_hip', lm[23]), 24: ('R_hip', lm[24]),
                            25: ('L_knee', lm[25]), 26: ('R_knee', lm[26]),
                            27: ('L_ankle', lm[27]), 28: ('R_ankle', lm[28]),
                        }
                    }
            except: pass
        return None

    def draw_pose_mesh(self, frame, landmarks):
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 23), (23, 24), (24, 25), (25, 27),
            (27, 31), (31, 29), (29, 27), (23, 24),
            (24, 26), (26, 28), (28, 32), (32, 30), (30, 28)
        ]
        for i, j in connections:
            if i < len(landmarks) and j < len(landmarks):
                x1 = int(landmarks[i].x * w)
                y1 = int(landmarks[i].y * h)
                x2 = int(landmarks[j].x * w)
                y2 = int(landmarks[j].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        return frame

    def draw_face_mesh(self, frame, landmarks):
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        return frame

    def init_yolo_pose(self):
        """Initialize YOLO-pose model for GPU-accelerated pose detection."""
        if hasattr(self, '_yolo_pose_model') and self._yolo_pose_model is not None:
            return True
        try:
            from ultralytics import YOLO
            import torch
            model_name = 'yolo11n-pose.pt'
            self._yolo_pose_model = YOLO(model_name)
            if torch.cuda.is_available():
                self._yolo_pose_model.to('cuda')
                log("YOLO-pose loaded with GPU acceleration")
            else:
                log("YOLO-pose loaded (CPU mode)")
            return True
        except Exception as e:
            log(f"YOLO-pose init failed: {e}")
            return False

    def detect_pose_yolo(self, bgr) -> dict | None:
        """
        Detect pose using YOLO-pose (GPU accelerated).
        Returns same format as detect_pose_full() for compatibility.
        """
        if not hasattr(self, '_yolo_pose_model') or self._yolo_pose_model is None:
            if not self.init_yolo_pose():
                return None
        try:
            results = self._yolo_pose_model.predict(
                bgr, conf=0.5, verbose=False, pose=True, device='cuda' if self._use_gpu else 'cpu'
            )
            if not results or len(results) == 0:
                return None
            r = results[0]
            if r.keypoints is None or len(r.keypoints) == 0:
                return None
            kpts = r.keypoints.data[0]
            if len(kpts) < 17:
                return None
            # COCO keypoints: 0=nose, 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
            # 9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip
            def make_lm(idx):
                k = kpts[idx]
                return type('obj', (object,), {'x': k[0].item(), 'y': k[1].item(), 'z': 0})()
            return {
                'nose': make_lm(0),
                'left_shoulder': make_lm(5), 'right_shoulder': make_lm(6),
                'left_elbow': make_lm(7), 'right_elbow': make_lm(8),
                'left_wrist': make_lm(9), 'right_wrist': make_lm(10),
                'left_hip': make_lm(11), 'right_hip': make_lm(12),
                'landmarks': [make_lm(i) for i in range(min(len(kpts), 17))]
            }
        except Exception as e:
            log(f"YOLO-pose error: {e}")
            return None

    def draw_yolo_pose(self, frame, result):
        """Draw pose skeleton from YOLO-pose result."""
        if result is None or 'landmarks' not in result:
            return frame
        h, w = frame.shape[:2]
        lm = result['landmarks']
        skeleton = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12)]
        for i, j in skeleton:
            if i < len(lm) and j < len(lm):
                x1, y1 = int(lm[i].x * w), int(lm[i].y * h)
                x2, y2 = int(lm[j].x * w), int(lm[j].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        for i in range(min(len(lm), 17)):
            x, y = int(lm[i].x * w), int(lm[i].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        return frame

# ── GripperCamera ──────────────────────────────────────────────────────────────
class GripperCamera:
    """
    Camera mounted on robot gripper for eye-in-hand vision.
    Streams video and provides frames for YOLOE object detection.
    """
    def __init__(self, cam_index=0, resolution=(1280, 720)):
        self.cam_index = cam_index
        self.resolution = resolution
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self._thread = None

        # YOLOE detector for object detection
        self.detector = None
        self._detector_lock = threading.Lock()

    def start(self):
        """Start the gripper camera."""
        if self.running:
            return
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.cam_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.running = True
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                log(f"Gripper camera started (cam {self.cam_index})")
            else:
                log(f"Failed to open gripper camera {self.cam_index}")
        except Exception as e:
            log(f"Gripper camera error: {e}")

    def _capture_loop(self):
        """Continuous frame capture."""
        import cv2
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            time.sleep(0.033)  # ~30 FPS

    def get_frame(self):
        """Get current frame."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop the gripper camera."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def load_yoloe(self, model_path=None):
        """Load YOLO model for object detection using ultralytics with GPU acceleration."""
        if model_path is None:
            model_path = os.path.expanduser("~/models/yoloe/yolo11n-seg.pt")
        if not os.path.exists(model_path):
            log(f"YOLO model not found at {model_path}")
            return False
        try:
            from ultralytics import YOLO
            self.detector = YOLO(model_path)
            # Enable GPU acceleration
            try:
                import torch
                if torch.cuda.is_available():
                    self.detector.to('cuda')
                    log("YOLO detector loaded with GPU acceleration (CUDA)")
                else:
                    log("YOLO detector loaded (GPU not available, using CPU)")
            except ImportError:
                log("YOLO detector loaded (PyTorch CUDA not available)")
            return True
        except Exception as e:
            log(f"YOLO load error: {e}")
            return False

    def detect_objects(self, frame=None, conf_threshold=0.25):
        """
        Detect objects using YOLO (via ultralytics).
        Returns list of detections with bounding boxes and classes.
        """
        if self.detector is None:
            return []

        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return []

        try:
            results = self.detector.predict(frame, conf=conf_threshold, verbose=False)
            detections = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })
            return detections
        except Exception as e:
            log(f"YOLO detection error: {e}")
            return []

    def get_detection_frame(self, show_detections=True):
        """
        Get frame with detection overlays.
        Used for streaming to UI.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        if show_detections and self.detector is not None:
            detections = self.detect_objects(frame)
            import cv2
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']}: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

# ── PID ───────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, lo, hi):
        self.kp, self.ki, self.kd, self.lo, self.hi = kp, ki, kd, lo, hi
        self._i = self._le = 0.0
        self._t = time.time()
        self.deadzone = 0.005

    def step(self, e: float) -> float:
        if abs(e) < self.deadzone: return 0.0
        dt = max(1e-4, time.time() - self._t)
        self._i += e * dt
        self._i = float(np.clip(self._i, self.lo / max(self.ki, 1e-6), self.hi / max(self.ki, 1e-6)))
        out = self.kp * e + self.ki * self._i + self.kd * (e - self._le) / dt
        self._le, self._t = e, time.time()
        return float(np.clip(out, self.lo, self.hi))

# ── TrackerThread ─────────────────────────────────────────────────────────────
class TrackerThread(threading.Thread):
    """Capture thread + async detection thread."""
    def __init__(self, robot, detector, cam_index: int, settings: dict):
        super().__init__(daemon=True)
        self.robot    = robot
        self.detector = detector
        self.cam      = cam_index
        self.s        = settings
        self.running  = False
        self.frame_q  = queue.Queue(maxsize=2)
        self._detect_in  = queue.Queue(maxsize=1)
        self._last_result = None
        self._result_lock = threading.Lock()
        self._stats_lock  = threading.Lock()
        self.stats = dict(fps=0.0, target_cx=0.5, target_cy=0.5,
                          pan=0.0, tilt=0.0, detected=False, hand_side="",
                          tracking_mode=settings.get("tracking_mode", "face"))
        self._pp = PID(settings["kp"], settings["ki"], settings["kd"],
                       -settings["pan_range"], settings["pan_range"])
        self._tp = PID(settings["kp"], settings["ki"], settings["kd"],
                       -settings["tilt_range"], settings["tilt_range"])

    def _detect_worker(self):
        mode = self.s.get("tracking_mode", "face")
        while self.running:
            try:
                small = self._detect_in.get(timeout=0.1)
            except queue.Empty:
                continue
            if mode == "hand":
                result = self.detector.detect_hand(small)
            elif mode == "pose":
                result = self.detector.detect_pose(small)
            elif mode == "body":
                result = self.detector.detect_pose_full(small)
            elif mode == "pose_detail":
                result = self.detector.detect_pose_detail(small)
            elif mode == "pose_yolo":
                result = self.detector.detect_pose_yolo(small)
            else:
                result = self.detector.detect_face(small)
            with self._result_lock:
                self._last_result = result

    def run(self):
        self.running = True
        threading.Thread(target=self._detect_worker, daemon=True).start()
        cap = cv2.VideoCapture(self.cam)
        if not cap.isOpened():
            log(f"Camera {self.cam} failed"); self.running = False; return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        pan  = float(self.s["pan_home"])
        tilt = float(self.s["tilt_home"])
        mode = self.s.get("tracking_mode", "face")
        t0 = time.time(); fc = 0
        try:
            while self.running:
                t_frame = time.time()
                ret, frame = cap.read()
                if not ret: time.sleep(0.01); continue
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (320, 240))
                try: self._detect_in.put_nowait(small)
                except queue.Full: pass
                with self._result_lock:
                    result = self._last_result
                if mode == "hand":
                    if result:
                        cx, cy, hand_side, hand_landmarks = result[:4]
                        pan = float((cx - 0.5) * self.s["pan_range"] * 2)
                        tilt = float((cy - 0.5) * self.s["tilt_range"] * 2)
                        pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                        tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                        self.robot.send_positions(
                            {"shoulder_pan": pan, "shoulder_lift": tilt,
                             "elbow_flex": self.s["elbow_flex"], "wrist_flex": self.s["wrist_flex"],
                             "wrist_roll": self.s["wrist_roll"], "gripper": self.s["gripper"]},
                            speed=1000, acc=100)
                        with self._stats_lock:
                            self.stats.update(pan=pan, tilt=tilt)
                        px, py = int(cx * w), int(cy * h)
                        color = (80, 200, 255) if hand_side == "Left" else (255, 150, 80)
                        cv2.circle(frame, (px, py), 22, color, 2)
                        cv2.line(frame, (px-30, py), (px+30, py), color, 1)
                        cv2.line(frame, (px, py-30), (px, py+30), color, 1)
                        if hand_landmarks:
                            frame = self.detector.draw_hand_mesh(frame, hand_landmarks, hand_side)
                        cv2.putText(frame, f"Hand {hand_side}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                        cv2.putText(frame, f"Pan:{pan:.1f} Tilt:{tilt:.1f}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False; self.stats["hand_side"] = ""
                        cv2.putText(frame, "No hand", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                elif mode == "pose":
                    if result:
                        cx, cy, pose_landmarks = result[:3]
                        with self._stats_lock:
                            self.stats.update(detected=True, target_cx=cx, target_cy=cy)
                        pan = float((cx - 0.5) * self.s["pan_range"] * 2)
                        tilt = float((cy - 0.5) * self.s["tilt_range"] * 2)
                        pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                        tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                        self.robot.send_positions(
                            {"shoulder_pan": pan, "shoulder_lift": tilt,
                             "elbow_flex": self.s["elbow_flex"], "wrist_flex": self.s["wrist_flex"],
                             "wrist_roll": self.s["wrist_roll"], "gripper": self.s["gripper"]},
                            speed=1000, acc=100)
                        with self._stats_lock:
                            self.stats.update(pan=pan, tilt=tilt)
                        px, py = int(cx * w), int(cy * h)
                        cv2.circle(frame, (px, py), 22, (255, 0, 255), 2)
                        cv2.line(frame, (px-30, py), (px+30, py), (255, 0, 255), 1)
                        cv2.line(frame, (px, py-30), (px, py+30), (255, 0, 255), 1)
                        if pose_landmarks:
                            frame = self.detector.draw_pose_mesh(frame, pose_landmarks)
                        cv2.putText(frame, f"Pose Pan:{pan:.1f} Tilt:{tilt:.1f}",
                                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False
                        cv2.putText(frame, "No pose", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                elif mode == "body":
                    if result:
                        left_shoulder = result.get('left_shoulder')
                        right_shoulder = result.get('right_shoulder')
                        left_elbow = result.get('left_elbow')
                        right_elbow = result.get('right_elbow')
                        left_wrist = result.get('left_wrist')
                        right_wrist = result.get('right_wrist')
                        nose = result.get('nose')
                        landmarks = result.get('landmarks')
                        with self._stats_lock:
                            self.stats.update(detected=True, target_cx=nose.x if nose else 0.5, target_cy=nose.y if nose else 0.5)
                        pan = float((nose.x - 0.5) * self.s["pan_range"] * 2) if nose else 0
                        tilt = float((nose.y - 0.5) * self.s["tilt_range"] * 2) if nose else 0
                        pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                        tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                        if left_elbow and left_wrist:
                            dx = left_wrist.x - left_elbow.x
                            dy = left_elbow.y - left_wrist.y
                            elbow_angle = float(np.clip(-np.sqrt(dx*dx + dy*dy) * 150, -120, 0))
                            wrist_angle = float(np.clip(dx * 90, -90, 90))
                        else:
                            elbow_angle = 0
                            wrist_angle = 0
                        self.robot.send_positions(
                            {"shoulder_pan": pan, "shoulder_lift": tilt,
                             "elbow_flex": elbow_angle, "wrist_flex": wrist_angle,
                             "wrist_roll": 0, "gripper": 0},
                            speed=1000, acc=100)
                        with self._stats_lock:
                            self.stats.update(pan=pan, tilt=tilt, elbow_flex=elbow_angle, wrist_flex=wrist_angle)
                        if landmarks:
                            frame = self.detector.draw_pose_mesh(frame, landmarks)
                        cv2.putText(frame, f"Body Pan:{pan:.1f} Tilt:{tilt:.1f} Elbow:{elbow_angle:.1f}",
                                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False
                        cv2.putText(frame, "No body", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                elif mode == "pose_detail":
                    if result and 'joints' in result:
                        lm = result['landmarks']
                        joints = result['joints']
                        joint_colors = {
                            0: ((0, 255, 255), 'Nose → Pan/Tilt'),  # Yellow
                            11: ((255, 100, 100), 'L_Shoulder'),  # Red
                            12: ((100, 100, 255), 'R_Shoulder'),  # Blue
                            13: ((255, 255, 100), 'L_Elbow → Elbow'),  # Cyan
                            14: ((100, 255, 255), 'R_Elbow'),  # Orange
                            15: ((100, 255, 100), 'L_Wrist → Wrist'),  # Green
                            16: ((255, 100, 255), 'R_Wrist'),  # Purple
                            23: ((200, 200, 200), 'L_Hip'),  # Gray
                            24: ((150, 150, 150), 'R_Hip'),  # Dark gray
                        }
                        for idx, (name, lm_obj) in joints.items():
                            x = int(lm_obj.x * w)
                            y = int(lm_obj.y * h)
                            col, label = joint_colors.get(idx, ((0, 255, 0), name))
                            cv2.circle(frame, (x, y), 10, col, -1)
                            cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)
                        for i, j in [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24)]:
                            if i < len(lm) and j < len(lm):
                                x1, y1 = int(lm[i].x * w), int(lm[i].y * h)
                                x2, y2 = int(lm[j].x * w), int(lm[j].y * h)
                                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                        l_shoulder = joints.get(11, (None, None))[1]
                        r_shoulder = joints.get(12, (None, None))[1]
                        l_elbow = joints.get(13, (None, None))[1]
                        l_wrist = joints.get(15, (None, None))[1]
                        nose = joints.get(0, (None, None))[1]
                        pan = float((nose.x - 0.5) * self.s["pan_range"] * 2) if nose else 0
                        tilt = float((nose.y - 0.5) * self.s["tilt_range"] * 2) if nose else 0
                        pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                        tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                        if l_elbow and l_wrist:
                            dx = l_wrist.x - l_elbow.x
                            dy = l_elbow.y - l_wrist.y
                            dist = np.sqrt(dx*dx + dy*dy)
                            elbow_angle = float(np.clip(-dist * 180, -120, 0))
                            wrist_angle = float(np.clip(dx * 90, -90, 90))
                        else:
                            elbow_angle = 0
                            wrist_angle = 0
                        self.robot.send_positions(
                            {"shoulder_pan": pan, "shoulder_lift": tilt,
                             "elbow_flex": elbow_angle, "wrist_flex": wrist_angle,
                             "wrist_roll": 0, "gripper": 0},
                            speed=1000, acc=100)
                        with self._stats_lock:
                            self.stats.update(detected=True, pan=pan, tilt=tilt, 
                                            elbow_flex=elbow_angle, wrist_flex=wrist_angle)
                        cv2.rectangle(frame, (5, 5), (320, 85), (0, 0, 0), -1)
                        cv2.rectangle(frame, (5, 5), (320, 85), (0, 255, 0), 2)
                        cv2.putText(frame, f"Pan: {pan:.0f}°  Tilt: {tilt:.0f}°", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(frame, f"Elbow: {elbow_angle:.0f}°  Wrist: {wrist_angle:.0f}°", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                        cv2.putText(frame, "Nose→Pan/Tilt | Arm→Elbow/Wrist", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                        y_legend = h - 60
                        cv2.putText(frame, "🟡 Nose", (8, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        cv2.putText(frame, "🟢 Arm", (100, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
                        cv2.putText(frame, "🔵 Wrist", (200, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                        cv2.putText(frame, "Move fast!", (8, y_legend + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False
                        cv2.putText(frame, "No pose detected", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                elif mode == "pose_yolo":
                    if result:
                        nose = result.get('nose')
                        landmarks = result.get('landmarks')
                        if nose:
                            cx = nose.x / w if hasattr(nose, 'x') else 0.5
                            cy = nose.y / h if hasattr(nose, 'y') else 0.5
                            with self._stats_lock:
                                self.stats.update(detected=True, target_cx=cx, target_cy=cy)
                            pan = float((cx - 0.5) * self.s["pan_range"] * 2)
                            tilt = float((cy - 0.5) * self.s["tilt_range"] * 2)
                            pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                            tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                            self.robot.send_positions(
                                {"shoulder_pan": pan, "shoulder_lift": tilt,
                                 "elbow_flex": self.s["elbow_flex"], "wrist_flex": self.s["wrist_flex"],
                                 "wrist_roll": self.s["wrist_roll"], "gripper": self.s["gripper"]},
                                speed=1000, acc=100)
                            with self._stats_lock:
                                self.stats.update(pan=pan, tilt=tilt)
                            px, py = int(nose.x), int(nose.y)
                            cv2.circle(frame, (px, py), 22, (255, 165, 0), 2)
                            cv2.line(frame, (px-30, py), (px+30, py), (255, 165, 0), 1)
                            cv2.line(frame, (px, py-30), (px, py+30), (255, 165, 0), 1)
                        if landmarks:
                            frame = self.detector.draw_yolo_pose(frame, result)
                        cv2.putText(frame, f"YOLO-Pose GPU Pan:{pan:.1f} Tilt:{tilt:.1f}",
                                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False
                        cv2.putText(frame, "No pose (YOLO)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                else:
                    if result:
                        cx, cy, landmarks = result[:3]
                        with self._stats_lock:
                            self.stats.update(detected=True, target_cx=cx, target_cy=cy)
                        pan = float((cx - 0.5) * self.s["pan_range"] * 2)
                        tilt = float((cy - 0.5) * self.s["tilt_range"] * 2)
                        pan = float(np.clip(pan, -self.s["pan_range"], self.s["pan_range"]))
                        tilt = float(np.clip(tilt, -self.s["tilt_range"], self.s["tilt_range"]))
                        self.robot.send_positions(
                            {"shoulder_pan": pan, "shoulder_lift": tilt,
                             "elbow_flex": self.s["elbow_flex"], "wrist_flex": self.s["wrist_flex"],
                             "wrist_roll": self.s["wrist_roll"], "gripper": self.s["gripper"]},
                            speed=1000, acc=100)
                        with self._stats_lock:
                            self.stats.update(pan=pan, tilt=tilt)
                        px, py = int(cx * w), int(cy * h)
                        cv2.circle(frame, (px, py), 22, (0, 255, 80), 2)
                        cv2.line(frame, (px-30, py), (px+30, py), (0, 255, 80), 1)
                        cv2.line(frame, (px, py-30), (px, py+30), (0, 255, 80), 1)
                        if landmarks:
                            frame = self.detector.draw_face_mesh(frame, landmarks)
                        cv2.putText(frame, f"Face Pan:{pan:.1f} Tilt:{tilt:.1f}",
                                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
                    else:
                        with self._stats_lock:
                            self.stats["detected"] = False
                        cv2.putText(frame, "No face", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 255), 2)
                cv2.drawMarker(frame, (w//2, h//2), (255, 220, 0), cv2.MARKER_CROSS, 24, 1)
                fc += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    with self._stats_lock:
                        self.stats["fps"] = fc / elapsed
                    fc = 0; t0 = time.time()
                try: self.frame_q.put_nowait(frame)
                except queue.Full: pass
                spent = time.time() - t_frame
                if spent < 1/30:
                    time.sleep(1/30 - spent)
        finally:
            cap.release()

    def stop(self): self.running = False
    def get_stats(self) -> dict:
        with self._stats_lock: return self.stats.copy()

# ── Port auto-detection ──────────────────────────────────────────────────
def auto_detect_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Check for typical USB-Serial descriptors
        if "USB" in p.description.upper() or "SERIAL" in p.description.upper() or "ACM" in p.device.upper() or "USB" in p.device.upper():
            log(f"Auto-detected port: {p.device} ({p.description})")
            return p.device
    
    # Fallback to first available port if none matched the description
    if ports:
        log(f"Falling back to first available port: {ports[0].device}")
        return ports[0].device
        
    return None

def auto_detect_camera() -> int:
    import cv2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in range(2):  # Only check 0 and 1
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.release()
                    log(f"Auto-detected camera: {idx}")
                    return idx
                cap.release()
            except Exception:
                pass
    return 0

def list_available_ports() -> list[str]:
    import glob
    return sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))

def list_available_cameras() -> list[dict]:
    import cv2
    import warnings
    cams = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in range(2):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cams.append({"index": idx, "width": int(w), "height": int(h), "deviceId": str(idx)})
                    cap.release()
            except Exception:
                pass
    if not cams:
        cams = [{"index": 0, "width": 640, "height": 480, "deviceId": "0"}]
    return cams

# ── TeleopThread ──────────────────────────────────────────────────────────────
class TeleopThread(threading.Thread):
    """Thread to run teleoperation scripts and capture their video output."""
    def __init__(self, mode: str, cam_index: int = 0, port: str = None, baudrate: int = 1000000):
        super().__init__(daemon=True)
        self.mode = mode
        self.cam_index = cam_index
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.process = None
        self._cap = None

    def run(self):
        self.running = True
        teleop_dir = Path(__file__).parent.parent / "teleoperation"
        script_map = {
            "sign": "sign_gesture.py",
            "onehand": "one_hand.py",
            "twohands": "two_hands.py"
        }
        script = script_map.get(self.mode)
        if not script:
            log(f"Unknown teleop mode: {self.mode}")
            return

        script_path = teleop_dir / script
        if not script_path.exists():
            log(f"Teleop script not found: {script_path}")
            return

        # Get robot port from state
        with _state_lock:
            robot_port = self.port or state['settings'].get('port', '/dev/ttyUSB0')
            robot_baud = state['settings'].get('baudrate', 1000000)

        # Start the teleoperation script for ROBOT CONTROL
        # Note: Script handles its own camera + MediaPipe and streams MJPEG to stdout
        env = os.environ.copy()
        env['TELEOP_PORT'] = robot_port
        env['TELEOP_BAUD'] = str(robot_baud)
        env['TELEOP_STREAM'] = '1'           # enable MJPEG-to-stdout mode
        env['TELEOP_CAM_INDEX'] = str(self.cam_index)
        env['PYTHONUNBUFFERED'] = '1'        # no stdout buffering on the pipe

        # Use conda Python (has correct NumPy)
        CONDA_PYTHON = "/home/parc/miniconda3/envs/py310_ml/bin/python"

        try:
            self.process = subprocess.Popen(
                [CONDA_PYTHON, '-u', str(script_path)],  # -u: unbuffered stdout
                cwd=str(teleop_dir),
                env=env,
                stdout=subprocess.PIPE,      # was DEVNULL — read_frames() needs this
                stderr=subprocess.PIPE
            )
            log(f"Started teleop script: {script}")
            
            # Log stderr in background
            def log_stderr():
                import threading
                def read_stderr():
                    try:
                        for line in self.process.stderr:
                            log(f"Teleop {script} stderr: {line.decode().strip()}")
                    except: pass
                threading.Thread(target=read_stderr, daemon=True).start()
            log_stderr()
        except Exception as e:
            log(f"Failed to start teleop script: {e}")

        self._cap = None

        # Read MJPEG from subprocess stdout — store raw JPEG bytes (no decode)
        self._stream_buf = b''
        BOUNDARY = b'--frame'

        def read_frames():
            buf = b''
            while self.running and self.process and self.process.poll() is None:
                try:
                    chunk = self.process.stdout.read(65536)
                    if not chunk:
                        time.sleep(0.005)
                        continue
                    buf += chunk
                    # Extract all complete MJPEG frames; keep only the latest
                    latest_jpeg = None
                    while True:
                        s = buf.find(BOUNDARY)
                        if s == -1:
                            break
                        e = buf.find(BOUNDARY, s + len(BOUNDARY))
                        if e == -1:
                            buf = buf[s:]   # keep incomplete tail
                            break
                        part = buf[s:e]
                        buf = buf[e:]
                        sep = part.find(b'\r\n\r\n')
                        if sep != -1:
                            jpeg = part[sep + 4:]
                            if len(jpeg) > 500:
                                latest_jpeg = jpeg  # discard older frames
                    if latest_jpeg is not None:
                        with self.frame_lock:
                            self.frame = latest_jpeg  # raw JPEG bytes
                except Exception:
                    break
        
        import threading
        threading.Thread(target=read_frames, daemon=True).start()

        # Keep thread alive while running
        while self.running:
            time.sleep(0.5)

        # Cleanup
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        log(f"Teleop stopped: {self.mode}")

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        log(f"Teleop stopped: {self.mode}")

    def get_frame(self):
        # Returns raw JPEG bytes or None — callers serve them directly (no re-encode)
        with self.frame_lock:
            return self.frame  # bytes, not numpy

# ── Global app state ──────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_auto_port = auto_detect_port()
_auto_cam = auto_detect_camera()
state = {
    "robot":           None,
    "connected":       False,
    "tracker":         None,
    "tracking_active": False,
    "calibrating":     False,
    "detector":        None,
    "gripper_cam":     None,
    "teleop":          None,
    "mode":            "idle",
    "settings": {
        "port":          _auto_port or "/dev/ttyUSB0",
        "baudrate":      1000000,
        "cam_index":     _auto_cam,
        "gripper_cam_index": 1,  # Usually second camera
        "pan_home":      0,   "tilt_home":   0,
        "pan_range":     45,  "tilt_range":  25,
        "elbow_flex":    0,   "wrist_flex":  0,
        "wrist_roll":    0,   "gripper":     0,
        "kp": 2.5,  "ki": 0.05,  "kd": 0.3,
        "speed": 500,  "acc": 50,
        "tracking_mode": "face",
    }
}

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.template_filter('t')
def t_filter(key, lang=None):
    return t(key, lang)

app.jinja_env.globals.update(t=t)

# ── MJPEG stream ──────────────────────────────────────────────────────────────
_PLACEHOLDER: bytes | None = None

def _get_placeholder() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        img = np.zeros((270, 480, 3), dtype=np.uint8)
        cv2.putText(img, "No stream", (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        _PLACEHOLDER = buf.tobytes() if ok and buf is not None else b''
    return _PLACEHOLDER

def _mjpeg_generator():
    while True:
        tracker = state.get("tracker")
        if tracker and tracker.running:
            try:
                frame = tracker.frame_q.get(timeout=0.05)
                disp  = cv2.resize(frame, (640, 480))
                _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                continue
            except queue.Empty:
                pass
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + _get_placeholder() + b'\r\n'
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gripper_cam/start', methods=['POST'])
def gripper_cam_start():
    """Start the gripper camera."""
    data = request.json or {}
    cam_index = data.get('cam_index', state['settings'].get('gripper_cam_index', 0))

    with _state_lock:
        if state['gripper_cam'] is None:
            state['gripper_cam'] = GripperCamera(cam_index=cam_index)
        state['gripper_cam'].start()

    return jsonify({'ok': True, 'cam_index': cam_index})

@app.route('/gripper_cam/stop', methods=['POST'])
def gripper_cam_stop():
    """Stop the gripper camera."""
    with _state_lock:
        if state['gripper_cam']:
            state['gripper_cam'].stop()
    return jsonify({'ok': True})

@app.route('/gripper_cam/frame')
def gripper_cam_frame():
    """Get single frame from gripper camera as JPEG."""
    gripper_cam = state.get('gripper_cam')
    if not gripper_cam or not gripper_cam.running:
        return jsonify({'error': 'Camera not running'}), 503

    frame = gripper_cam.get_detection_frame(show_detections=True)
    if frame is None:
        return jsonify({'error': 'No frame'}), 503

    import cv2
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/gripper_cam/video_feed')
def gripper_cam_video_feed():
    """MJPEG stream from gripper camera - raw feed without detection for speed."""
    _enc = [cv2.IMWRITE_JPEG_QUALITY, 75]
    placeholder = _get_placeholder()
    def generate():
        while True:
            t0 = time.monotonic()
            gripper_cam = state.get('gripper_cam')
            if gripper_cam and gripper_cam.running:
                frame = gripper_cam.get_frame()
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame, _enc)
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                    elapsed = time.monotonic() - t0
                    remaining = 0.033 - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                    continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n'
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gripper_cam/detect', methods=['POST'])
def gripper_cam_detect():
    """Detect objects using YOLOE on gripper camera feed."""
    gripper_cam = state.get('gripper_cam')
    if not gripper_cam or not gripper_cam.running:
        return jsonify({'error': 'Camera not running'}), 503

    data = request.json or {}
    conf_threshold = data.get('conf', 0.25)

    # Load YOLOE if not loaded
    if gripper_cam.detector is None:
        gripper_cam.load_yoloe()

    detections = gripper_cam.detect_objects(conf_threshold=conf_threshold)

    return jsonify({
        'detections': detections,
        'count': len(detections)
    })

@app.route('/gripper_cam/status')
def gripper_cam_status():
    """Get gripper camera status."""
    gripper_cam = state.get('gripper_cam')
    return jsonify({
        'running': gripper_cam.running if gripper_cam else False,
        'has_detector': gripper_cam.detector is not None if gripper_cam else False
    })

# ── SSE: logs ─────────────────────────────────────────────────────────────────
@app.route('/stream/logs')
def stream_logs():
    q: queue.Queue = queue.Queue(maxsize=200)
    with _log_lock:
        _log_subs.append(q)
        recent = list(_log_entries[-40:])

    def generate():
        for entry in recent:
            yield f'data: {json.dumps(entry)}\n\n'
        try:
            while True:
                try:
                    entry = q.get(timeout=30)
                    yield f'data: {json.dumps(entry)}\n\n'
                except queue.Empty:
                    yield ': keepalive\n\n'
        finally:
            with _log_lock:
                try: _log_subs.remove(q)
                except: pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ── SSE: stats ────────────────────────────────────────────────────────────────
@app.route('/stream/stats')
def stream_stats():
    def generate():
        while True:
            tracker  = state.get("tracker")
            detector = state.get("detector")
            ts = tracker.get_stats() if (tracker and tracker.running) else {}
            data = {
                "connected":       state["connected"],
                "tracking_active": state["tracking_active"],
                "calibrating":     state["calibrating"],
                "calibrated":      calibration.calibrated,
                "mode":            state["mode"],
                "backend":         detector.backend if detector else "none",
                **ts,
            }
            yield f'data: {json.dumps(data)}\n\n'
            time.sleep(0.15)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ── REST API ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', 
                           servo_ids=SERVO_IDS, 
                           settings=state["settings"], 
                           lang=_current_lang, 
                           en_json=_translations.get("en", {}),
                           fr_json=_translations.get("fr", {}))

@app.route('/learn')
def learn():
    return render_template('pages/learn.html',
                           lang=_current_lang,
                           en_json=_translations.get("en", {}),
                           fr_json=_translations.get("fr", {}))

@app.route('/play')
def play():
    return render_template('pages/play.html',
                           lang=_current_lang,
                           en_json=_translations.get("en", {}),
                           fr_json=_translations.get("fr", {}))

@app.route('/settings')
def settings():
    return render_template('pages/settings.html',
                           lang=_current_lang,
                           en_json=_translations.get("en", {}),
                           fr_json=_translations.get("fr", {}))

# ── Language ───────────────────────────────────────────────────────────────────
@app.before_request
def read_lang_cookie():
    global _current_lang
    lang = request.cookies.get('lang', 'en')
    if lang in _translations:
        _current_lang = lang

@app.route('/api/lang', methods=['GET', 'POST'])
def api_lang():
    global _current_lang
    if request.method == 'POST':
        data = request.json or {}
        lang = data.get('lang', 'en')
        if lang in _translations:
            _current_lang = lang
            return jsonify({'ok': True, 'lang': _current_lang})
        return jsonify({'ok': False, 'msg': 'Invalid language'})
    return jsonify({'lang': _current_lang, 'available': list(_translations.keys())})

# ── Status ────────────────────────────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    tracker = state.get("tracker")
    ts = tracker.get_stats() if (tracker and tracker.running) else {}
    return jsonify({
        "connected":       state["connected"],
        "tracking_active": state["tracking_active"],
        "calibrated":      calibration.calibrated,
        "mode":            state["mode"],
        "settings":        state["settings"],
        "servo_ids":       SERVO_IDS,
        "tracker_stats":   ts,
    })

# ── Connection ────────────────────────────────────────────────────────────────
@app.route('/api/connect', methods=['POST'])
def api_connect():
    data = request.json or {}
    port     = data.get('port',     state['settings']['port'])
    baudrate = int(data.get('baudrate', state['settings']['baudrate']))
    with _state_lock:
        if state['robot']: state['robot'].disconnect()
        robot = Robot(port, baudrate)
        ok, msg = robot.connect()
        state['robot']    = robot if ok else None
        state['connected'] = ok
        state['settings']['port']     = port
        state['settings']['baudrate'] = baudrate
    return jsonify({'ok': ok, 'msg': msg})

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    with _state_lock:
        if state['robot']: state['robot'].disconnect()
        state['robot']     = None
        state['connected'] = False
    return jsonify({'ok': True})

@app.route('/api/robot/state', methods=['GET'])
def api_robot_state():
    """
    Get current robot state: joint angles (degrees) and FK position.
    Used for real-time sync between real robot and 3D simulation.

    Returns:
        {
            connected: bool,
            joints: {joint_name: angle_deg},
            raw: {joint_name: servo_value},
            fk_position: [x, y, z] in meters,
            fk_rotation: 3x3 matrix
        }
    """
    with _state_lock:
        robot = state.get('robot')
        connected = state.get('connected', False)

    if not connected or not robot:
        return jsonify({
            'connected': False,
            'joints': {},
            'raw': {},
            'fk_position': [0, 0, 0],
            'fk_rotation': [[1,0,0],[0,1,0],[0,0,1]]
        })

    try:
        robot_state = robot.get_joint_states()
        robot_state['connected'] = True
        return jsonify(robot_state)
    except Exception as e:
        log(f"Error reading robot state: {e}")
        return jsonify({
            'connected': False,
            'error': str(e)
        }), 500

@app.route('/api/robot/joints', methods=['POST'])
def api_robot_joints():
    """
    Send target joint positions to robot (for 3D-to-robot sync).
    Body: {'joints': {'shoulder_pan': 45, 'shoulder_lift': 30, ...}}
    """
    data = request.json or {}
    joints = data.get('joints', {})

    if not state.get('connected') or not state.get('robot'):
        return jsonify({'ok': False, 'msg': 'Not connected'})

    try:
        ok = state['robot'].send_positions(joints)
        return jsonify({'ok': ok})
    except Exception as e:
        log(f"Error sending joints: {e}")
        return jsonify({'ok': False, 'msg': str(e)}), 500

@app.route('/api/sim/ik', methods=['POST'])
def api_sim_ik():
    """
    Compute Inverse Kinematics for target position.
    Body: {'position': [x, y, z], 'rotation': [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]], 'initial_joints': {...}}
    Returns: {'joints': {...}, 'error': float, 'success': bool}
    """
    data = request.json or {}
    # accept both 'position' and legacy 'target' key
    position = data.get('position') or data.get('target', [0, 0, 0.3])
    rotation = data.get('rotation')
    initial_joints = data.get('initial_joints')

    # Default initial_joints to current robot state for faster convergence
    if initial_joints is None and state['connected'] and state.get('robot'):
        try:
            robot = state['robot']
            js = robot.get_joint_states()
            if js:
                initial_joints = js['joints']
        except Exception:
            pass

    result = Robot.compute_ik(position, rotation, initial_joints)
    return jsonify(result)

@app.route('/api/sim/trajectory', methods=['POST'])
def api_sim_trajectory():
    """
    Generate smooth trajectory between joint configurations using cubic splines.
    Body: {
        'waypoints': [
            {'joints': {'shoulder_pan': 0, ...}, 'duration': 2.0},
            {'joints': {'shoulder_pan': 45, ...}, 'duration': 2.0},
            ...
        ]
    }
    Returns: {'trajectory': [{'joints': {...}, 'time': float}, ...], 'total_duration': float}
    """
    data = request.json or {}
    waypoints = data.get('waypoints', [])

    if len(waypoints) < 2:
        return jsonify({'error': 'At least 2 waypoints required'}), 400

    trajectory = []
    current_time = 0

    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        duration = end.get('duration', 2.0)

        start_joints = start.get('joints', {})
        end_joints = end.get('joints', {})

        num_steps = max(int(duration * 50), 10)  # 50 Hz

        for step in range(num_steps + 1):
            t = step / num_steps
            t2 = t * t
            t3 = t2 * t

            # Cubic Hermite spline coefficients
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2

            point = {}
            for joint in joint_names:
                p0 = start_joints.get(joint, 0)
                p1 = end_joints.get(joint, 0)
                m0 = 0  # Zero derivative at waypoints
                m1 = 0
                point[joint] = round(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1, 2)

            trajectory.append({
                'joints': point,
                'time': round(current_time + t * duration, 3)
            })

        current_time += duration

    return jsonify({
        'trajectory': trajectory,
        'total_duration': round(current_time, 3)
    })

@app.route('/api/sim/move_to', methods=['POST'])
def api_sim_move_to():
    """
    Plan and execute move to target position using IK and trajectories.
    Body: {
        'target': [x, y, z],  # Target position in meters
        'gripper': 50,         # Gripper position
        'duration': 2.0        # Move duration
    }
    """
    data = request.json or {}
    target = data.get('target', [0, 0, 0.3])
    gripper = data.get('gripper', 50)
    duration = data.get('duration', 2.0)

    # Get current position as initial guess for IK
    current_state = None
    if state.get('connected') and state.get('robot'):
        try:
            current_state = state['robot'].get_joint_states()
        except:
            pass

    initial_joints = None
    if current_state and current_state.get('joints'):
        initial_joints = current_state['joints']

    # Solve IK
    ik_result = Robot.compute_ik(target, initial_joints=initial_joints)

    if not ik_result.get('success'):
        return jsonify({
            'ok': False,
            'error': ik_result.get('error', 'IK failed'),
            'joints': ik_result.get('joints')
        }), 400

    # Add gripper to solution
    ik_result['joints']['gripper'] = gripper

    # Generate trajectory
    waypoints = [
        {'joints': ik_result['joints'], 'duration': duration}
    ]

    trajectory_result = {}
    if current_state and current_state.get('joints'):
        trajectory_result = {
            'trajectory': [
                {'joints': current_state['joints'], 'time': 0},
                {'joints': ik_result['joints'], 'time': duration}
            ],
            'total_duration': duration
        }
    else:
        # Simple direct move
        trajectory_result = {
            'trajectory': [
                {'joints': ik_result['joints'], 'time': duration}
            ],
            'total_duration': duration
        }

    return jsonify({
        'ok': True,
        'joints': ik_result['joints'],
        'fk_position': (Robot.compute_fk(ik_result['joints'])[0].tolist()),
        'trajectory': trajectory_result.get('trajectory', [])
    })

@app.route('/api/sim/scene', methods=['POST'])
def api_sim_scene():
    """
    Set up a pick and place scene with objects.
    Body: {
        'objects': [{'name': 'red_cube', 'color': 'red', 'shape': 'cube', 'position': [x, y, z]}, ...],
        'bin': {'position': [x, y, z]}
    }
    Returns: List of detected objects with positions
    """
    data = request.json or {}

    # This is handled client-side in the 3D viewer
    # Just return success and store scene data
    scene_data = {
        'objects': data.get('objects', []),
        'bin': data.get('bin', {'position': [0.1, 0, 0]})
    }

    return jsonify({
        'ok': True,
        'scene': scene_data,
        'message': 'Scene data received. Use /api/sim/vision_frame to render.'
    })

@app.route('/api/sim/vision_frame', methods=['GET'])
def api_sim_vision_frame():
    """
    Get simulated vision frame from 3D scene.
    Returns: Base64 encoded JPEG image
    """
    # Render is done client-side
    return jsonify({
        'message': 'Vision frame rendering is done client-side via RobotArm3D.renderVisionFrame()'
    })

@app.route('/api/sim/objects', methods=['GET'])
def api_sim_objects():
    """
    Get list of objects in the simulation scene.
    Returns: [{'name': 'red_cube', 'color': 'red', 'position': [x, y, z]}, ...]
    """
    # This data is stored client-side
    return jsonify({
        'objects': [],
        'message': 'Object tracking is done client-side'
    })

@app.route('/api/vision/detect_scene', methods=['POST'])
def api_vision_detect_scene():
    """
    Detect objects in the simulation scene using YOLOE.
    Body: {
        'image': base64_image (optional, will render if not provided)
    }
    Returns: Detected objects with bounding boxes and positions
    """
    data = request.json or {}

    # For simulation, we return the known object positions
    # In real mode, this would use YOLOE on the camera feed
    objects = data.get('objects', [
        {'name': 'red_cube', 'color': 'red', 'position': [-0.05, 0.05, 0.025]},
        {'name': 'blue_cube', 'color': 'blue', 'position': [0.02, -0.03, 0.025]},
        {'name': 'green_cube', 'color': 'green', 'position': [0.05, 0.08, 0.025]}
    ])

    # Convert to detection format
    detections = []
    for i, obj in enumerate(objects):
        detections.append({
            'class': obj.get('color', 'unknown') + '_cube',
            'confidence': 0.95,
            'bbox': [0.1 + i * 0.15, 0.1, 0.1, 0.1],  # Normalized bbox
            'position_3d': obj.get('position', [0, 0, 0]),
            'name': obj.get('name', f'object_{i}')
        })

    return jsonify({
        'detections': detections,
        'count': len(detections),
        'source': 'simulation'
    })

@app.route('/api/ai/plan_pick_place', methods=['POST'])
def api_ai_plan_pick_place():
    """
    AI planning endpoint for pick and place.
    Uses RAG to get relevant knowledge and generates execution plan.
    Body: {
        'command': 'pick the red cube and put it in the bin',
        'detected_objects': [...],
        'scene': {...}
    }
    """
    data = request.json or {}
    command = data.get('command', '')
    detected_objects = data.get('detected_objects', [])
    scene = data.get('scene', {})

    # Find target object in detected objects
    target_color = None
    if 'red' in command.lower():
        target_color = 'red'
    elif 'blue' in command.lower():
        target_color = 'blue'
    elif 'green' in command.lower():
        target_color = 'green'

    # Find the object
    target_obj = None
    for obj in detected_objects:
        if target_color and target_color in obj.get('class', '').lower():
            target_obj = obj
            break

    if not target_obj:
        return jsonify({
            'ok': False,
            'error': f'Could not find {target_color} cube in scene',
            'command': command
        }), 400

    # Get bin position (default or from scene)
    bin_position = scene.get('bin', {}).get('position', [0.1, 0, 0.05])

    # Generate plan using Tucker kinematics
    pick_pos = target_obj.get('position_3d', [0, 0, 0.025])
    pre_pick_pos = [pick_pos[0], pick_pos[1], max(pick_pos[2] + 0.05, 0.05)]
    post_pick_pos = [pick_pos[0], pick_pos[1], 0.1]
    pre_place_pos = [bin_position[0], bin_position[1], 0.1]
    place_pos = [bin_position[0], bin_position[1], bin_position[2] + 0.025]

    # Compute IK for each position
    plan = {
        'steps': [
            {
                'action': 'move_to',
                'position': pre_pick_pos,
                'gripper': 50,
                'duration': 2.0,
                'description': 'Move above red cube'
            },
            {
                'action': 'move_to',
                'position': pick_pos,
                'gripper': 50,
                'duration': 1.0,
                'description': 'Descend to red cube'
            },
            {
                'action': 'grip',
                'gripper': 5,
                'duration': 0.5,
                'description': 'Close gripper on cube'
            },
            {
                'action': 'move_to',
                'position': post_pick_pos,
                'gripper': 5,
                'duration': 1.0,
                'description': 'Lift cube'
            },
            {
                'action': 'move_to',
                'position': pre_place_pos,
                'gripper': 5,
                'duration': 2.0,
                'description': 'Move above bin'
            },
            {
                'action': 'move_to',
                'position': place_pos,
                'gripper': 5,
                'duration': 1.0,
                'description': 'Descend into bin'
            },
            {
                'action': 'release',
                'gripper': 50,
                'duration': 0.5,
                'description': 'Release cube into bin'
            }
        ],
        'target_object': target_obj,
        'bin_position': bin_position
    }

    return jsonify({
        'ok': True,
        'command': command,
        'plan': plan,
        'simulation_ready': True
    })

# ── Calibration ──────────────────────────────────────────────────────────────
@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    data = request.json or {}
    limits_deg = data.get('limits')  # optional: {servo_name: {min, ctr, max}} in degrees
    def _run():
        state['calibrating'] = True
        try:
            state['robot'].auto_calibrate(limits_deg=limits_deg)
        finally:
            state['calibrating'] = False
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'msg': 'Calibration started (watch logs)'})

@app.route('/api/calibrate/stop', methods=['POST'])
def api_calibrate_stop():
    calibration._stop_event.set()
    state['calibrating'] = False
    log("⏹ Stop calibration requested")
    return jsonify({'ok': True, 'msg': 'Calibration arrêtée'})

@app.route('/api/calibrate/save', methods=['POST'])
def api_calibrate_save():
    data = request.json or {}
    limits = data.get('limits')  # {servo_name: {min_deg, ctr_deg, max_deg}} from UI table

    if limits:
        # Build from UI servo limits (degrees)
        calibration.min_deg    = {k: float(v['min']) for k, v in limits.items()}
        calibration.center_deg = {k: float(v['ctr']) for k, v in limits.items()}
        calibration.max_deg    = {k: float(v['max']) for k, v in limits.items()}
        for k, v in limits.items():
            inv = SERVO_INVERT.get(k, 1)
            calibration.min_pos[k]    = max(0, min(4095, int(2048 + float(v['min']) * inv * DEGREE_TO_UNIT)))
            calibration.center_pos[k] = max(0, min(4095, int(2048 + float(v['ctr']) * inv * DEGREE_TO_UNIT)))
            calibration.max_pos[k]    = max(0, min(4095, int(2048 + float(v['max']) * inv * DEGREE_TO_UNIT)))
        calibration.calibrated = True
    elif not calibration.calibrated:
        return jsonify({'ok': False, 'msg': 'Aucune calibration à sauvegarder'})

    save_data = {
        'limits_deg': {k: {'min': calibration.min_deg.get(k, -90),
                           'ctr': calibration.center_deg.get(k, 0),
                           'max': calibration.max_deg.get(k, 90)}
                       for k in SERVO_IDS},
         'min_pos':    calibration.min_pos,
         'max_pos':    calibration.max_pos,
         'center_pos': calibration.center_pos,
         'home_pos':   calibration.center_pos.copy(),
    }
    cal_file = Path(__file__).parent / 'calibration.json'
    with open(cal_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"✔ Calibration saved → {cal_file.name}")
    return jsonify({'ok': True, 'msg': 'Calibration sauvegardée', 'limits_deg': save_data['limits_deg']})

@app.route('/api/calibrate/load', methods=['POST'])
def api_calibrate_load():
    cal_file = Path(__file__).parent / 'calibration.json'
    if not cal_file.exists():
        return jsonify({'ok': False, 'msg': 'Fichier calibration.json introuvable'})
    try:
        with open(cal_file) as f:
            data = json.load(f)
        calibration.min_pos    = data.get('min_pos',    {})
        calibration.max_pos    = data.get('max_pos',    {})
        calibration.center_pos = data.get('center_pos', {})
        calibration.home_pos   = data.get('home_pos', calibration.center_pos.copy())
        calibration.calibrated = bool(calibration.center_pos)
        limits_deg = data.get('limits_deg')
        if limits_deg:
            calibration.min_deg    = {k: v['min'] for k, v in limits_deg.items()}
            calibration.center_deg = {k: v['ctr'] for k, v in limits_deg.items()}
            calibration.max_deg    = {k: v['max'] for k, v in limits_deg.items()}
        elif calibration.center_pos:
            # Reconstruct degrees from raw units
            calibration.min_deg    = {k: round((calibration.min_pos.get(k, 1548) - 2048) / DEGREE_TO_UNIT, 1) for k in calibration.center_pos}
            calibration.center_deg = {k: round((calibration.center_pos.get(k, 2048) - 2048) / DEGREE_TO_UNIT, 1) for k in calibration.center_pos}
            calibration.max_deg    = {k: round((calibration.max_pos.get(k, 2548) - 2048) / DEGREE_TO_UNIT, 1) for k in calibration.center_pos}
            limits_deg = {k: {'min': calibration.min_deg[k], 'ctr': calibration.center_deg[k], 'max': calibration.max_deg[k]} for k in calibration.center_pos}
        log(f"✔ Calibration loaded from {cal_file.name}")
        return jsonify({'ok': True, 'msg': 'Calibration chargée', 'limits_deg': limits_deg})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route('/api/calibrate/reset', methods=['POST'])
def api_calibrate_reset():
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    if not calibration.calibrated or not calibration.center_deg:
        return jsonify({'ok': False, 'msg': 'Pas de calibration chargée'})
    spd = state['settings'].get('speed', 500)
    ac  = state['settings'].get('acc', 50)
    state['robot'].send_positions(calibration.center_deg, speed=spd, acc=ac)
    log("↺ Servos → positions calibrées (center)")
    return jsonify({'ok': True, 'msg': 'Servos repositionnés sur la calibration'})

@app.route('/api/calibrate/set_center', methods=['POST'])
def api_calibrate_set_center():
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    data = request.json or {}
    servo = data.get('servo')
    if servo and servo in SERVO_IDS:
        current = state['robot'].read_raw_positions()
        if servo in current:
            calibration.center_pos[servo] = current[servo]
            calibration.center_deg[servo] = round((current[servo] - 2048) / DEGREE_TO_UNIT, 1)
            log(f"Set {servo} center to {current[servo]} ({calibration.center_deg[servo]}°)")
            cal_file = Path(__file__).parent / 'calibration.json'
            save_data = {
                'limits_deg': {k: {'min': calibration.min_deg.get(k, -90),
                                   'ctr': calibration.center_deg.get(k, 0),
                                   'max': calibration.max_deg.get(k, 90)}
                               for k in SERVO_IDS},
                'min_pos':    calibration.min_pos,
                'max_pos':    calibration.max_pos,
                'center_pos': calibration.center_pos,
            }
            with open(cal_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            return jsonify({'ok': True, 'msg': f'{servo} center set to {current[servo]} ({calibration.center_deg[servo]}°)', 'center_deg': calibration.center_deg})
    return jsonify({'ok': False, 'msg': 'Invalid servo name'})

# ── Tracking ──────────────────────────────────────────────────────────────────
@app.route('/api/tracking/start', methods=['POST'])
def api_tracking_start():
    data = request.json or {}
    with _state_lock:
        s = {**state['settings'], **data}
        state['settings'].update(data)
        if state['tracker']:
            state['tracker'].stop()
            time.sleep(0.3)
        if state['detector'] is None:
            state['detector'] = FaceDetector()
        robot = state['robot']
        if not robot or not robot.connected:
            robot = DummyRobot()
        tr = TrackerThread(robot, state['detector'], int(s.get('cam_index') or 0), state['settings'])
        tr.start()
        state['tracker']         = tr
        state['tracking_active'] = True
    return jsonify({'ok': True})

@app.route('/api/ports', methods=['GET'])
def api_list_ports():
    return jsonify({"ports": list_available_ports()})

@app.route('/api/cameras', methods=['GET'])
def api_list_cameras():
    return jsonify({"cameras": list_available_cameras()})

@app.route('/api/tracking/stop', methods=['POST'])
def api_tracking_stop():
    with _state_lock:
        if state['tracker']: state['tracker'].stop()
        state['tracking_active'] = False
    return jsonify({'ok': True})

# ── Teleoperation ────────────────────────────────────────────────────────────
@app.route('/api/teleop/run', methods=['POST'])
def api_teleop_run():
    """
    Run a teleoperation Python script directly.
    """
    import subprocess
    data = request.json or {}
    mode = data.get('mode', 'onehand')
    port = data.get('port', '/dev/ttyUSB0')
    
    script_map = {
        "sign": "sign_gesture.py",
        "onehand": "one_hand.py", 
        "twohands": "two_hands.py"
    }
    
    script = script_map.get(mode)
    if not script:
        return jsonify({'ok': False, 'error': 'Unknown mode'})
    
    script_path = Path(__file__).parent.parent / "teleoperation" / script
    if not script_path.exists():
        return jsonify({'ok': False, 'error': f'Script not found: {script}'})
    
    # Kill any existing teleop process
    with _state_lock:
        if state.get('teleop_process'):
            try:
                state['teleop_process'].terminate()
            except:
                pass
            state['teleop_process'] = None
    
    # Run via conda environment - use direct path to conda python
    script_dir = "/home/parc/parc_final/teleoperation"
    CONDA_PYTHON = "/home/parc/miniconda3/envs/py310_ml/bin/python"
    
    env = os.environ.copy()
    env['TELEOP_PORT'] = port
    
    try:
        # Directly call the conda env's python
        proc = subprocess.Popen(
            [CONDA_PYTHON, script],
            cwd=script_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        with _state_lock:
            state['teleop_process'] = proc
        log(f"Started teleop script: {script} on port {port}")
        return jsonify({'ok': True, 'mode': mode, 'pid': proc.pid})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/teleop/start', methods=['POST'])
def api_teleop_start():
    data = request.json or {}
    mode = data.get('mode', 'onehand')
    cam_index = int(data.get('cam_index', 0))

    with _state_lock:
        if state['teleop']:
            state['teleop'].stop()
            time.sleep(1.0)   # give OS time to release the camera fd
        teleop = TeleopThread(mode, cam_index)
        teleop.start()
        state['teleop'] = teleop
    return jsonify({'ok': True, 'mode': mode})

@app.route('/api/teleop/stop', methods=['POST'])
def api_teleop_stop():
    with _state_lock:
        # Kill subprocess
        if state.get('teleop_process'):
            try:
                state['teleop_process'].terminate()
                state['teleop_process'].wait(timeout=2)
            except:
                try:
                    state['teleop_process'].kill()
                except: pass
            state['teleop_process'] = None
        # Also stop TeleopThread if running
        if state.get('teleop'):
            state['teleop'].stop()
            state['teleop'] = None
    return jsonify({'ok': True})

@app.route('/teleop/video_feed')
def teleop_video_feed():
    placeholder = _get_placeholder()
    def generate():
        while True:
            t0 = time.monotonic()
            with _state_lock:
                teleop = state.get('teleop')
            jpeg = (teleop.get_frame() if teleop and teleop.running else None) or placeholder
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            elapsed = time.monotonic() - t0
            remaining = 0.033 - elapsed  # target ~30 fps
            if remaining > 0:
                time.sleep(remaining)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Settings ─────────────────────────────────────────────────────────────────
@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json or {}
    global SERVO_INVERT
    with _state_lock:
        state['settings'].update(data)
        if 'servo_ids' in data:
            SERVO_IDS.update(data['servo_ids'])
            with open(Path(__file__).parent / 'servo_ids.json', 'w') as f:
                json.dump(SERVO_IDS, f)
        if 'servo_invert' in data:
            SERVO_INVERT.update(data['servo_invert'])
            with open(Path(__file__).parent / 'servo_invert.json', 'w') as f:
                json.dump(SERVO_INVERT, f)
    return jsonify({'ok': True})

# ── Servo ────────────────────────────────────────────────────────────────────
@app.route('/api/servo/move', methods=['POST'])
def api_servo_move():
    data = request.json or {}
    robot = state.get('robot')
    if not state['connected'] or not robot or not robot.connected:
        return jsonify({'ok': False, 'msg': 'Robot non connecté'})
    name    = data.get('name')
    pos_deg = float(data.get('pos_deg', 0))
    spd     = int(data.get('speed', 300))
    ac      = int(data.get('acc', 30))
    sid     = SERVO_IDS.get(name)
    if sid is None:
        return jsonify({'ok': False, 'msg': f'Servo inconnu: {name}'})
    inv = SERVO_INVERT.get(name, 1)
    pos_unit = max(0, min(4095, int(2048 + pos_deg * inv * DEGREE_TO_UNIT)))
    ok = robot.write_raw_position(sid, pos_unit, speed=spd, acc=ac, invert=inv)
    if ok:
        log(f"Servo {name} (ID{sid}): {pos_deg:.1f}° → {pos_unit} units")
        _TRACKER_FIXED = {'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'}
        if name in _TRACKER_FIXED:
            state['settings'][name] = pos_deg
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'msg': f'Écriture échouée pour {name} (ID{sid})'})

@app.route('/api/servo/hw_limits', methods=['POST'])
def api_servo_hw_limits():
    """Lit les limites angulaires min/max stockées dans l'EEPROM du servo."""
    data = request.json or {}
    robot = state.get('robot')
    if not state['connected'] or not robot or not robot.connected:
        return jsonify({'ok': False, 'msg': 'Robot non connecté'})
    name = data.get('name')
    sid  = SERVO_IDS.get(name)
    if sid is None:
        return jsonify({'ok': False, 'msg': f'Servo inconnu: {name}'})
    try:
        with robot._lock:
            raw_min, res_min, _ = robot.servo.read2ByteTxRx(sid, 9)
            raw_max, res_max, _ = robot.servo.read2ByteTxRx(sid, 11)
            mode_val, res_mode, _ = robot.servo.read1ByteTxRx(sid, 33)  # SMS_STS_MODE
        if res_min != COMM_SUCCESS or res_max != COMM_SUCCESS:
            return jsonify({'ok': False, 'msg': f'Lecture EEPROM échouée (ID{sid})'})
        deg_min  = round((raw_min - 2048) / DEGREE_TO_UNIT, 1)
        deg_max  = round((raw_max - 2048) / DEGREE_TO_UNIT, 1)
        mode_str = {0: 'position', 1: 'wheel (vitesse)', 2: 'step'}.get(mode_val, f'inconnu({mode_val})')
        log(f"HW limits {name} (ID{sid}): raw {raw_min}–{raw_max} → {deg_min}°–{deg_max}° | MODE={mode_val} ({mode_str})")
        return jsonify({'ok': True, 'name': name, 'raw_min': raw_min, 'raw_max': raw_max,
                        'deg_min': deg_min, 'deg_max': deg_max,
                        'mode': mode_val, 'mode_str': mode_str})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route('/api/servo/wheel_speed', methods=['POST'])
def api_servo_wheel_speed():
    """Contrôle la vitesse d'un servo en mode roue (wheel mode). speed: -1023..+1023, 0=stop."""
    data = request.json or {}
    robot = state.get('robot')
    if not state['connected'] or not robot or not robot.connected:
        return jsonify({'ok': False, 'msg': 'Robot non connecté'})
    name = data.get('name')
    sid  = SERVO_IDS.get(name)
    if sid is None:
        return jsonify({'ok': False, 'msg': f'Servo inconnu: {name}'})
    speed = int(data.get('speed', 0))
    acc   = int(data.get('acc', 50))
    try:
        with robot._lock:
            robot.servo.WriteSpec(sid, speed, acc)
        log(f"WheelSpeed {name} (ID{sid}): speed={speed}")
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route('/api/servo/read', methods=['POST'])
def api_servo_read():
    data = request.json or {}
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    name = data.get('name')
    sid  = SERVO_IDS.get(name)
    if sid is None:
        return jsonify({'ok': False, 'msg': f'Unknown servo: {name}'})
    try:
        with state['robot']._lock:
            pos, result, _ = state['robot'].servo.ReadPos(sid)
        if result == COMM_SUCCESS and pos is not None:
            deg = (pos - 2048) / DEGREE_TO_UNIT
            log(f"Read {name}: {int(pos)} = {deg:.1f}°")
            return jsonify({'ok': True, 'pos': int(pos), 'deg': round(deg, 1)})
        return jsonify({'ok': False, 'msg': 'Read failed'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route('/api/servo/read_all', methods=['GET'])
def api_servo_read_all():
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected', 'positions': {}})
    positions = {}
    try:
        with state['robot']._lock:
            for name, sid in SERVO_IDS.items():
                pos, result, _ = state['robot'].servo.ReadPos(sid)
                if result == COMM_SUCCESS and pos is not None:
                    deg = (pos - 2048) / DEGREE_TO_UNIT
                    positions[name] = round(deg, 1)
        return jsonify({'ok': True, 'positions': positions})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e), 'positions': {}})

@app.route('/api/mode/run', methods=['POST'])
def api_mode_run():
    data = request.json or {}
    mode = data.get('mode', state['mode'])
    state['mode'] = mode
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    threading.Thread(target=_execute_mode, args=(mode,), daemon=True).start()
    return jsonify({'ok': True, 'msg': f'Running {mode}'})

@app.route('/api/quick_action', methods=['POST'])
def api_quick_action():
    data   = request.json or {}
    action = data.get('action')
    if not state['connected'] or not state['robot']:
        return jsonify({'ok': False, 'msg': 'Not connected'})
    HOME = {"shoulder_pan": 0, "shoulder_lift": 88, "elbow_flex": -85,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 0}
    spd = state['settings'].get('speed', 500)
    ac  = state['settings'].get('acc', 50)
    if   action == 'home': state['robot'].send_positions(HOME, speed=spd, acc=ac)
    elif action == 'wave': state['robot'].send_positions(
        {"shoulder_pan": 25, "shoulder_lift": -30, "elbow_flex": 50,
         "wrist_flex": 10, "wrist_roll": 50, "gripper": -60}, speed=spd, acc=ac)
    elif action == 'grip': state['robot'].send_positions({**HOME, "gripper": 60}, speed=spd, acc=ac)
    log(f"Quick action: {action}")
    return jsonify({'ok': True})

@app.route('/robot-arm')
def robot_arm():
    return render_template('robot_arm_2d_visualizer.html',
                           lang=_current_lang,
                           en_json=_translations.get("en", {}),
                           fr_json=_translations.get("fr", {}))

@app.route('/api/move_all', methods=['POST'])
def api_move_all():
    data = request.json or {}
    robot = state.get('robot')
    if not state['connected'] or not robot or not robot.connected:
        return jsonify({'ok': False, 'msg': 'Robot not connected'})

    # Map from UI names to servo names (handle both formats for compatibility)
    angles = {
        "shoulder_pan":  float(data.get('shoulder_pan', data.get('pan', 0))),
        "shoulder_lift": float(data.get('shoulder_lift', data.get('tilt', 0))),
        "elbow_flex":    float(data.get('elbow_flex', data.get('elbow', 0))),
        "wrist_flex":    float(data.get('wrist_flex', data.get('wrist_flex', 0))),
        "wrist_roll":    float(data.get('wrist_roll', data.get('wrist_roll', 0))),
        "gripper":       float(data.get('gripper', 0))
    }

    spd = int(data.get('speed', 500))
    acc = int(data.get('acc', 50))

    ok = robot.send_positions(angles, speed=spd, acc=acc)
    return jsonify({'ok': ok})

# ── Mode execution ────────────────────────────────────────────────────────────
def _execute_mode(mode_name: str):
    HOME = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": 0}
    spd = state['settings'].get('speed', 500)
    ac  = state['settings'].get('acc',   50)

    def run_steps(name, steps, times=1):
        log(f"▶ {name}")
        robot = state.get('robot')
        if not robot: return
        for t in range(times):
            if times > 1: log(f"  loop {t+1}/{times}")
            for i, step in enumerate(steps):
                label = step[0]; pos = step[1]
                d  = step[2] if len(step) > 2 else 0.9
                sp = step[3] if len(step) > 3 else spd
                ac2 = step[4] if len(step) > 4 else ac
                robot.send_positions(pos, speed=sp, acc=ac2)
                log(f"  [{i+1}/{len(steps)}] {label}")
                time.sleep(d)
        log(f"✔ {name} done")

    if mode_name == "idle":
        run_steps("Idle", [
            ("Settle", {**HOME, "shoulder_lift": 10, "elbow_flex": -10}, 0.6, 300, 20),
            ("Rest",   {**HOME},                                          0.5, 200, 15),
        ])

    elif mode_name == "hello":
        WB = {"shoulder_pan": 25, "shoulder_lift": -30, "elbow_flex": 50,
              "wrist_flex": 10, "wrist_roll": 0, "gripper": -20}
        run_steps("Hello", [
            ("Raise arm",  {**WB},                               0.5, 900, 70),
            ("Open",       {**WB, "gripper": -60},               0.25, 600, 50),
            ("Wave R",     {**WB, "wrist_roll":  55},            0.22, 800, 65),
            ("Wave L",     {**WB, "wrist_roll": -55},            0.22, 800, 65),
            ("Wave R",     {**WB, "wrist_roll":  55},            0.22, 800, 65),
            ("Wave L",     {**WB, "wrist_roll": -55},            0.22, 800, 65),
            ("Wave R",     {**WB, "wrist_roll":  55},            0.22, 800, 65),
            ("Wave L",     {**WB, "wrist_roll": -55},            0.22, 800, 65),
            ("Lower",      {**WB, "shoulder_lift": 0,
                            "elbow_flex": 20, "wrist_roll": 0},  0.4, 600, 50),
            ("Home",       {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 60}, 0.5, 500, 40),
        ])

    elif mode_name == "pick_place":
        run_steps("Pick & Place", [
            ("Above pick", {"shoulder_pan": -35, "shoulder_lift": -20,
                            "elbow_flex":  30, "wrist_flex": 20, "wrist_roll": 0, "gripper": -50}, 0.6, 800, 65),
            ("Descend",    {"shoulder_pan": -35, "shoulder_lift":  15,
                            "elbow_flex": -35, "wrist_flex": 15, "wrist_roll": 0, "gripper": -50}, 0.6, 500, 40),
            ("Grip",       {"shoulder_pan": -35, "shoulder_lift":  15,
                            "elbow_flex": -35, "wrist_flex": 15, "wrist_roll": 0, "gripper":  40}, 0.7, 350, 30),
            ("Lift",       {"shoulder_pan": -35, "shoulder_lift": -25,
                            "elbow_flex":  35, "wrist_flex": 10, "wrist_roll": 0, "gripper":  40}, 0.5, 900, 70),
            ("Carry",      {"shoulder_pan":  35, "shoulder_lift": -20,
                            "elbow_flex":  30, "wrist_flex": 10, "wrist_roll": 0, "gripper":  40}, 0.6, 800, 65),
            ("Lower",      {"shoulder_pan":  35, "shoulder_lift":  15,
                            "elbow_flex": -35, "wrist_flex": 15, "wrist_roll": 0, "gripper":  40}, 0.6, 500, 40),
            ("Release",    {"shoulder_pan":  35, "shoulder_lift":  15,
                            "elbow_flex": -35, "wrist_flex": 15, "wrist_roll": 0, "gripper": -50}, 0.7, 350, 30),
            ("Clear",      {"shoulder_pan":  35, "shoulder_lift": -20,
                            "elbow_flex":  30, "wrist_flex":  0, "wrist_roll": 0, "gripper": -50}, 0.4, 900, 70),
            ("Home",       {**HOME},                                                                0.5, 600, 50),
        ])

    elif mode_name == "dance":
        B = 0.35
        run_steps("Dance", [
            ("Ready",     {**HOME, "shoulder_lift": -10},                                              B,   600, 50),
            ("Left",      {"shoulder_pan": -45, "shoulder_lift": -15, "elbow_flex":  25,
                           "wrist_flex":  0, "wrist_roll":  30, "gripper": -20},                     B,   800, 60),
            ("Right",     {"shoulder_pan":  45, "shoulder_lift": -15, "elbow_flex":  25,
                           "wrist_flex":  0, "wrist_roll": -30, "gripper": -20},                     B,   800, 60),
            ("Up pump",   {"shoulder_pan":   0, "shoulder_lift": -40, "elbow_flex":  55,
                           "wrist_flex": 15, "wrist_roll":   0, "gripper": -30},                     B,   800, 60),
            ("Down pump", {"shoulder_pan":   0, "shoulder_lift":  25, "elbow_flex": -40,
                           "wrist_flex":-10, "wrist_roll":   0, "gripper":  20},                     B,   800, 60),
            ("Twist L",   {"shoulder_pan": -30, "shoulder_lift":  -5, "elbow_flex":  10,
                           "wrist_flex":  5, "wrist_roll":  70, "gripper": -20},                     B,   800, 60),
            ("Twist R",   {"shoulder_pan":  30, "shoulder_lift":  -5, "elbow_flex":  10,
                           "wrist_flex":  5, "wrist_roll": -70, "gripper": -20},                     B,   800, 60),
            ("Flourish",  {"shoulder_pan":  20, "shoulder_lift": -35, "elbow_flex":  50,
                           "wrist_flex": 20, "wrist_roll":  45, "gripper": -50},                     B*2, 700, 50),
            ("Reset",     {**HOME, "shoulder_lift": -10},                                              B,   600, 50),
        ], times=3)

    elif mode_name == "stretch":
        run_steps("Stretch", [
            ("Forward",     {"shoulder_pan":   0, "shoulder_lift": -40, "elbow_flex":  60,
                             "wrist_flex":  25, "wrist_roll":   0, "gripper": -30}, 0.9, 700, 55),
            ("Left reach",  {"shoulder_pan": -60, "shoulder_lift": -25, "elbow_flex":  40,
                             "wrist_flex":  10, "wrist_roll": -30, "gripper": -30}, 0.9, 700, 55),
            ("High center", {"shoulder_pan":   0, "shoulder_lift": -45, "elbow_flex":  65,
                             "wrist_flex":  30, "wrist_roll":   0, "gripper": -30}, 0.8, 700, 55),
            ("Right reach", {"shoulder_pan":  60, "shoulder_lift": -25, "elbow_flex":  40,
                             "wrist_flex":  10, "wrist_roll":  30, "gripper": -30}, 0.9, 700, 55),
            ("Wrist L",     {"shoulder_pan":  10, "shoulder_lift": -20, "elbow_flex":  30,
                             "wrist_flex":  10, "wrist_roll":  80, "gripper": -20}, 0.6, 600, 50),
            ("Wrist R",     {"shoulder_pan":  10, "shoulder_lift": -20, "elbow_flex":  30,
                             "wrist_flex":  10, "wrist_roll": -80, "gripper": -20}, 0.6, 600, 50),
            ("Relax",       {"shoulder_pan":   0, "shoulder_lift":   5, "elbow_flex": -15,
                             "wrist_flex":  -5, "wrist_roll":   0, "gripper":   0}, 0.8, 400, 30),
            ("Home",        {**HOME},                                                0.5, 500, 40),
        ])

    elif mode_name == "salute":
        run_steps("Salute", [
            ("Raise", {"shoulder_pan":  15, "shoulder_lift": -30, "elbow_flex":  55,
                       "wrist_flex":  30, "wrist_roll":  20, "gripper": -10}, 0.5, 900, 70),
            ("Hold",  {"shoulder_pan":  15, "shoulder_lift": -30, "elbow_flex":  55,
                       "wrist_flex":  30, "wrist_roll":  20, "gripper": -10}, 1.2, 200, 10),
            ("Snap",  {"shoulder_pan":  10, "shoulder_lift":  -5, "elbow_flex":  15,
                       "wrist_flex":   5, "wrist_roll":   0, "gripper":   0}, 0.3, 900, 80),
            ("Home",  {**HOME},                                                0.5, 600, 50),
        ])

    elif mode_name == "shake":
        SK = {"shoulder_pan": -20, "shoulder_lift": 5, "elbow_flex": -20,
              "wrist_flex": 0, "wrist_roll": 0, "gripper": -50}
        run_steps("Handshake", [
            ("Extend",  {**SK},                                     0.7, 450, 30),
            ("Open",    {**SK, "gripper": -60},                     0.3, 300, 20),
            ("Close",   {**SK, "gripper":  30},                     0.3, 350, 25),
            ("Shake 1", {**SK, "wrist_flex":  20, "gripper": 30},   0.2, 700, 55),
            ("Shake 2", {**SK, "wrist_flex": -20, "gripper": 30},   0.2, 700, 55),
            ("Shake 3", {**SK, "wrist_flex":  20, "gripper": 30},   0.2, 700, 55),
            ("Shake 4", {**SK, "wrist_flex": -20, "gripper": 30},   0.2, 700, 55),
            ("Release", {**SK, "wrist_flex":   0, "gripper": -50},  0.4, 400, 30),
            ("Withdraw",{**HOME, "shoulder_lift": -10},              0.5, 450, 30),
            ("Home",    {**HOME},                                    0.5, 300, 20),
        ])

# ── Llama CPP Integration ───────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).parent))
from llama_integration import (
    get_ai_assistant, 
    get_vision_assistant, 
    get_audio_assistant,
    setup_ai_routes
)

# Setup all AI routes
setup_ai_routes(app)

# ── Simulation server process ─────────────────────────────────────────────────
_SIM_SERVER = Path(__file__).parent.parent / "simulation" / "server.py"
_SIM_HTTP_PORT = int(os.environ.get("SIM_HTTP_PORT", 38000))
_sim_proc: subprocess.Popen | None = None
_sim_lock = threading.Lock()


def _sim_running() -> bool:
    return _sim_proc is not None and _sim_proc.poll() is None


@app.route('/api/simulation/status', methods=['GET'])
def api_simulation_status():
    running = _sim_running()
    return jsonify({'running': running, 'url': f'http://localhost:{_SIM_HTTP_PORT}' if running else None})


@app.route('/api/simulation/start', methods=['POST'])
def api_simulation_start():
    global _sim_proc
    with _sim_lock:
        if _sim_running():
            return jsonify({'ok': True, 'already_running': True, 'url': f'http://localhost:{_SIM_HTTP_PORT}'})
        if not _SIM_SERVER.exists():
            return jsonify({'ok': False, 'error': f'server.py not found at {_SIM_SERVER}'})
        try:
            _sim_proc = subprocess.Popen(
                [sys.executable, str(_SIM_SERVER)],
                cwd=str(_SIM_SERVER.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log(f"Simulation server started (pid {_sim_proc.pid})")
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True, 'url': f'http://localhost:{_SIM_HTTP_PORT}'})


@app.route('/api/simulation/stop', methods=['POST'])
def api_simulation_stop():
    global _sim_proc
    with _sim_lock:
        if not _sim_running():
            return jsonify({'ok': True, 'was_running': False})
        try:
            _sim_proc.terminate()
            _sim_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _sim_proc.kill()
        log(f"Simulation server stopped")
        _sim_proc = None
    return jsonify({'ok': True, 'was_running': True})


# ── MuJoCo viewer (native desktop window) ────────────────────────────────────
_VIEWER_SCRIPT = Path(__file__).parent.parent / "simulation" / "viewer.py"
_viewer_proc: subprocess.Popen | None = None
_viewer_lock = threading.Lock()


def _viewer_running() -> bool:
    return _viewer_proc is not None and _viewer_proc.poll() is None


@app.route('/api/viewer/status', methods=['GET'])
def api_viewer_status():
    return jsonify({'running': _viewer_running()})


@app.route('/api/viewer/start', methods=['POST'])
def api_viewer_start():
    global _viewer_proc
    with _viewer_lock:
        if _viewer_running():
            return jsonify({'ok': True, 'already_running': True})
        if not _VIEWER_SCRIPT.exists():
            return jsonify({'ok': False, 'error': f'viewer.py not found at {_VIEWER_SCRIPT}'})
        try:
            # miniconda base has mujoco 3.8; py310_ml does not — use base
            viewer_python = '/home/parc/miniconda3/bin/python'
            env = os.environ.copy()
            env.setdefault('DISPLAY', ':1')
            env['SO101_DIR'] = str(Path(__file__).parent.parent / 'simulation' / 'so101-inverse-kinematics-main' / 'so101')
            # PYTHONPATH inherits the system python3.10 user-site path which
            # breaks numpy ABI under Python 3.13 — clear it for this subprocess
            env.pop('PYTHONPATH', None)
            _viewer_proc = subprocess.Popen(
                [viewer_python, str(_VIEWER_SCRIPT), '--controller', 'http://127.0.0.1:5000'],
                cwd=str(_VIEWER_SCRIPT.parent),
                env=env,
                stderr=subprocess.PIPE,
            )
            log(f"MuJoCo viewer started (pid {_viewer_proc.pid}) via {viewer_python}")
            # log stderr in background so errors surface in the webapp log
            def _log_viewer_stderr():
                for line in _viewer_proc.stderr:
                    log(f"[viewer] {line.decode().rstrip()}")
            threading.Thread(target=_log_viewer_stderr, daemon=True).start()
            # Move real robot to the viewer launch stance
            _VIEWER_INIT_STANCE = {
                "shoulder_pan":  0,
                "shoulder_lift": 80,
                "elbow_flex":   -89,
                "wrist_flex":    0,
                "wrist_roll":    90,
                "gripper":       0,
            }
            robot = state.get('robot')
            if state.get('connected') and robot:
                try:
                    robot.send_positions(_VIEWER_INIT_STANCE)
                    log("Viewer launch: real robot moved to initial stance")
                except Exception as re:
                    log(f"Viewer launch: could not move robot: {re}")
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True})


@app.route('/api/viewer/stop', methods=['POST'])
def api_viewer_stop():
    global _viewer_proc
    with _viewer_lock:
        if not _viewer_running():
            return jsonify({'ok': True, 'was_running': False})
        try:
            _viewer_proc.terminate()
            _viewer_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _viewer_proc.kill()
        log("MuJoCo viewer stopped")
        _viewer_proc = None
    return jsonify({'ok': True, 'was_running': True})


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("PARC ROBOTICS - SO-101 Robot Controller")
    print("=" * 60)
    print("Web UI: http://localhost:5000")
    print("AI Assistant: Ensure llama.cpp server is running on port 8080")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
