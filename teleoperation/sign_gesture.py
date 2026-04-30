import sys
import os
from pathlib import Path
# Prepend the custom SDK so sms_sts is available regardless of system packages
sys.path.insert(0, str(Path(__file__).parent.parent / "control_arm" / "stservo-env"))

import cv2
import mediapipe as mp
import math
import numpy as np
import time

# --- STREAMING MODE ---
STREAM_MODE = os.environ.get('TELEOP_STREAM', '0') == '1'
NO_CAMERA = os.environ.get('TELEOP_NOCALIB', '0') == '1'

# In stream mode stdout is binary MJPEG — redirect print() to stderr
if STREAM_MODE:
    import builtins as _builtins
    _real_print = _builtins.print
    def _stderr_print(*args, **kwargs):
        kwargs.setdefault('file', sys.stderr)
        _real_print(*args, **kwargs)
    _builtins.print = _stderr_print
TARGET_FPS = 30
_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

def stream_frame(frame):
    """Output frame as MJPEG to stdout"""
    _, buf = cv2.imencode('.jpg', frame, _ENCODE_PARAMS)
    sys.stdout.buffer.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
    sys.stdout.buffer.write(buf.tobytes())
    sys.stdout.buffer.write(b'\r\n')
    sys.stdout.buffer.flush()

# ── 1. CONFIGURATION ──
PORT = os.environ.get('TELEOP_PORT', '/dev/ttyACM0')
BAUD = int(os.environ.get('TELEOP_BAUD', '1000000'))
SERVO_IDS = {"pan": 1, "tilt": 2, "elbow": 3, "flex": 4, "rot": 5, "grip": 6}
LIMITS = {
    "pan": (1024, 3072), "tilt": (1024, 3072), "elbow": (1024, 3072),
    "flex": (1024, 3072), "rot": (1024, 3072), "grip": (1800, 3000)
}
INITIAL_STANCE = {1: 1935, 2: 2013, 3: 1741, 4: 2957, 5: 2028, 6: 2030}

# CONTROL SENSITIVITY (Lower = Slower/Smoother)
SENSITIVITY_XY = 1500  # Pixels to Servo Units scaling
ALPHA_SMOOTH = 0.15    # Lowered for better vertical stability

class SO101Robot:
    def __init__(self, port_name, baud_rate):
        self.last_sent_pos = INITIAL_STANCE.copy()
        self.connected = False
        self.ref_hand = None    # Where the hand started
        self.ref_servo = None   # Where the servo was at start
        
        try:
            from scservo_sdk import PortHandler, sms_sts
            self.port = PortHandler(port_name)
            if self.port.openPort() and self.port.setBaudRate(baud_rate):
                self.packet = sms_sts(self.port)
                self.connected = True
                self.reset_to_stance()
        except: print("⚠️ Robot Offline - Simulation Mode")

    def reset_to_stance(self):
        for s_id, pos in INITIAL_STANCE.items():
            self.send_raw(s_id, pos)
            self.last_sent_pos[s_id] = pos
            time.sleep(0.05)

    def send_raw(self, s_id, pos):
        if self.connected:
            self.packet.WritePosEx(s_id, int(pos), 300, 50)

    def update_relative(self, key, current_hand_coord, is_new_gesture):
        s_id = SERVO_IDS[key]
        
        # 1. Reset Reference on first detection
        if is_new_gesture or self.ref_hand is None:
            self.ref_hand = current_hand_coord
            self.ref_servo = self.last_sent_pos[s_id]
            return

        # 2. Calculate Delta (Difference from start)
        # Note: Vertical (y) is inverted so Up = Positive Delta
        delta = (self.ref_hand - current_hand_coord) * SENSITIVITY_XY
        
        # 3. Apply to starting position
        target = self.ref_servo + delta
        lo, hi = LIMITS[key]
        target = np.clip(target, lo, hi)
        
        # 4. Smooth and Send
        smooth_val = (ALPHA_SMOOTH * target) + (1 - ALPHA_SMOOTH) * self.last_sent_pos[s_id]
        self.last_sent_pos[s_id] = smooth_val
        
        # Keep ALL servos under torque
        for id_h, pos_h in self.last_sent_pos.items():
            self.send_raw(id_h, pos_h)

def get_sign_gesture(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]
    
    if up[0] and up[3] and not up[1]: return "WRIST"
    if up[0] and up[1] and not up[2]: return "TILT"
    if up[0] and not any(up[1:]):    return "PAN"
    if up[3] and not any(up[:3]):    return "ELBOW"
    if math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.04: return "GRIP_OPEN"
    return "IDLE"

def main():
    robot = SO101Robot(PORT, BAUD)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    cam_idx = int(os.environ.get('TELEOP_CAM_INDEX', '0'))
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Auto-activate when launched from webapp; require keyboard 'b' in local mode
    is_active = STREAM_MODE
    last_gesture = "IDLE"
    _frame_dt = 1.0 / TARGET_FPS

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8) as hands:
        while cap.isOpened():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_axis = "LOCKED"

            # Design tokens (PARC palette — BGR)
            GREEN  = (129, 185, 16)   # #10b981
            AMBER  = (35,  166, 245)  # #F5A623
            DARK   = (23,  14,  10)   # #0a0e17
            MUTED  = (139, 116, 100)  # #64748b
            WHITE  = (220, 204, 196)  # #c4ccdc

            if res.multi_hand_landmarks and is_active:
                for hand_lms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    lm = hand_lms.landmark
                    gesture = get_sign_gesture(lm)
                    current_axis = gesture
                    
                    is_new = (gesture != last_gesture)
                    
                    # Vertical Tracking (Y) - Tilt, Elbow, Wrist
                    if gesture in ["TILT", "ELBOW", "WRIST"]:
                        # Use specific landmark for verticality
                        target_y = lm[8].y if gesture == "TILT" else (lm[20].y if gesture == "ELBOW" else lm[0].y)
                        robot.update_relative(gesture.lower() if gesture != "WRIST" else "flex", target_y, is_new)
                    
                    # Horizontal Tracking (X) - Pan
                    elif gesture == "PAN":
                        robot.update_relative("pan", -lm[8].x, is_new) # Inverted x for natural mirror
                    
                    elif gesture == "GRIP_OPEN":
                        robot.send_raw(SERVO_IDS["grip"], LIMITS["grip"][0])
                        robot.last_sent_pos[6] = LIMITS["grip"][0]
                    else:
                        # Auto-close and maintain torque
                        robot.send_raw(SERVO_IDS["grip"], LIMITS["grip"][1])
                        robot.last_sent_pos[6] = LIMITS["grip"][1]
                        # Refresh all other servos to hold position
                        for s_id, pos in robot.last_sent_pos.items(): robot.send_raw(s_id, pos)

                    last_gesture = gesture
            else:
                last_gesture = "IDLE"
                robot.ref_hand = None

            # ── PARC-style minimal overlay ────────────────────────────────────

            # Bottom status bar (semi-transparent strip)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 44), (w, h), DARK, -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

            # Status pill
            status_txt  = "ACTIVE" if is_active else "LOCKED"
            status_col  = GREEN if is_active else AMBER
            pill_w = 80
            cv2.rectangle(frame, (10, h - 36), (10 + pill_w, h - 10), status_col, -1)
            cv2.putText(frame, status_txt, (14, h - 17),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, DARK, 1, cv2.LINE_AA)

            # Gesture label (centre)
            g_label = current_axis if current_axis != "LOCKED" else ("---" if is_active else "")
            if g_label:
                tw = cv2.getTextSize(g_label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0][0]
                cv2.putText(frame, g_label, (w // 2 - tw // 2, h - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

            # Hint text right side
            hint = "B:start  R:reset" if not STREAM_MODE else ""
            cv2.putText(frame, hint, (w - 160, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)

            # Compact legend (top-left corner) — only visible when no hand or locked
            hand_present = bool(res.multi_hand_landmarks)
            if not is_active or not hand_present:
                legend = [
                    ("INDEX finger", "Pan"),
                    ("PEACE  v-sign", "Tilt"),
                    ("PINKY  alone",  "Elbow"),
                    ("HORNS  \\m/",    "Wrist"),
                    ("PINCH  O",       "Grip open"),
                ]
                lx, ly = 10, 16
                lw_box, lh_box = 188, len(legend) * 18 + 22
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (lx, ly), (lx + lw_box, ly + lh_box), DARK, -1)
                cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
                cv2.putText(frame, "SIGN CONTROL", (lx + 6, ly + 14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, GREEN, 1, cv2.LINE_AA)
                for i, (gesture_name, axis) in enumerate(legend):
                    y = ly + 30 + i * 18
                    cv2.putText(frame, gesture_name, (lx + 6, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1, cv2.LINE_AA)
                    cv2.putText(frame, axis, (lx + 130, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

            # ─────────────────────────────────────────────────────────────────

            # Output frame - stream to stdout or display locally
            if STREAM_MODE:
                stream_frame(frame)
            else:
                cv2.imshow('SO101 SIGN relative-control', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('b'): is_active = True
                elif key == ord('r'):
                    is_active = False
                    robot.reset_to_stance()
                elif key == ord('q'): break

            # FPS cap — sleep the remainder of the frame budget
            elapsed = time.monotonic() - t0
            remaining = _frame_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    if not STREAM_MODE:
        cv2.destroyAllWindows()
    else:
        sys.stdout.buffer.flush()

if __name__ == "__main__": main()
