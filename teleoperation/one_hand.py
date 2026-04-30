import sys
import os
from pathlib import Path
# Add SDK path
SDK_PATH = Path(__file__).parent.parent / "control_arm" / "stservo-env"
sys.path.insert(0, str(SDK_PATH))

import cv2
import mediapipe as mp
import numpy as np
from scservo_sdk import PortHandler, sms_sts
import sys
import os
import time

# --- STREAMING MODE ---
STREAM_MODE = os.environ.get('TELEOP_STREAM', '0') == '1'
NO_CAMERA = os.environ.get('TELEOP_NOCALIB', '0') == '1'
TARGET_FPS = 30
_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

# In stream mode stdout is binary MJPEG — redirect print() to stderr
if STREAM_MODE:
    import builtins as _builtins
    _real_print = _builtins.print
    def _stderr_print(*args, **kwargs):
        kwargs.setdefault('file', sys.stderr)
        _real_print(*args, **kwargs)
    _builtins.print = _stderr_print

def stream_frame(frame):
    """Output frame as MJPEG to stdout"""
    _, buf = cv2.imencode('.jpg', frame, _ENCODE_PARAMS)
    sys.stdout.buffer.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
    sys.stdout.buffer.write(buf.tobytes())
    sys.stdout.buffer.write(b'\r\n')
    sys.stdout.buffer.flush()

# --- CONFIGURATION ---
import os
PORT = os.environ.get('TELEOP_PORT', '/dev/ttyACM0')
BAUD = int(os.environ.get('TELEOP_BAUD', '1000000'))
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42

INITIAL_STANCE = {
    1: 1935, 2: 2013, 3: 1741, 
    4: 2957, 5: 2028, 6: 2030
}

# Paramètres de contrôle
SMOOTHING = 0.15 
PINCH_THRESHOLD = 0.05  
FIST_THRESHOLD = 0.12   

# --- ZONE DE CIBLE AGRANDIE (HAUTEUR AUGMENTÉE) ---
BOX_X_MIN, BOX_X_MAX = 0.30, 0.70  # Largeur
BOX_Y_MIN, BOX_Y_MAX = 0.15, 0.85  # Hauteur augmentée (de 15% à 85% de l'écran)

class RoboticArm:
    def __init__(self):
        self.port = PortHandler(PORT)
        self.is_tracking = False

        if not (self.port.openPort() and self.port.setBaudRate(BAUD)):
            print("[ERR] Connexion échouée.")
            exit()

        self.packet = sms_sts(self.port)  # must be created after port open

        for s_id in INITIAL_STANCE.keys():
            self.packet.write1ByteTxRx(s_id, ADDR_TORQUE_ENABLE, 1)

        self.current_pos = INITIAL_STANCE.copy()
        self.reset_to_initial()

    def move_servo(self, s_id, pos):
        pos = int(np.clip(pos, 0, 4095))
        self.packet.WritePosEx(s_id, pos, 300, 50)

    def reset_to_initial(self):
        print("[ACTION] Réinitialisation...")
        self.is_tracking = False 
        for s_id, pos in INITIAL_STANCE.items():
            self.move_servo(s_id, pos)
            self.current_pos[s_id] = pos

    def update_active_dofs(self, id1_target, id2_target, is_pinched):
        if not self.is_tracking: return
        self.current_pos[1] = int((id1_target * SMOOTHING) + (self.current_pos[1] * (1 - SMOOTHING)))
        self.current_pos[2] = int((id2_target * SMOOTHING) + (self.current_pos[2] * (1 - SMOOTHING)))
        self.current_pos[6] = 3000 if is_pinched else 2030
        self.move_servo(1, self.current_pos[1])
        self.move_servo(2, self.current_pos[2])
        self.move_servo(6, self.current_pos[6])

def main():
    arm = RoboticArm()
    # Auto-activate tracking when launched from webapp (stream mode)
    if STREAM_MODE:
        arm.is_tracking = True

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cam_idx = int(os.environ.get('TELEOP_CAM_INDEX', '0'))
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last_wrist_pos = None
    _frame_dt = 1.0 / TARGET_FPS

    while cap.isOpened():
        t0 = time.monotonic()
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # --- 1. BARRE LATÉRALE ---
        cv2.rectangle(frame, (0, 0), (220, h), (30, 30, 30), -1) 
        cv2.putText(frame, "CONTROLES", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, "[B] Toggle Track", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[R] Reset Stance", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[Q] Quitter", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        status_color = (0, 255, 0) if arm.is_tracking else (0, 0, 255)
        status_text = "ACTIF" if arm.is_tracking else "BLOQUE"
        cv2.putText(frame, f"STATUS: {status_text}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # --- 2. DESSIN DU FRAME ET RÉTICULE ---
        bx1, by1 = int(BOX_X_MIN * w), int(BOX_Y_MIN * h)
        bx2, by2 = int(BOX_X_MAX * w), int(BOX_Y_MAX * h)
        cx, cy = int(w // 2), int(h // 2)
        
        box_color = (0, 255, 0) if arm.is_tracking else (0, 0, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
        
        # Crosshair
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), box_color, 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), box_color, 1)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
                
                # Indicateurs de position si bloqué
                if not arm.is_tracking:
                    if wrist.y < BOX_Y_MIN:
                        cv2.putText(frame, "TROP HAUT", (cx - 60, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif wrist.y > BOX_Y_MAX:
                        cv2.putText(frame, "TROP BAS", (cx - 60, by2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif BOX_X_MIN < wrist.x < BOX_X_MAX:
                        cv2.putText(frame, "OK! APPUYEZ SUR [B]", (cx - 100, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Reset par poing
                tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
                avg_dist = sum([np.sqrt((hand_lms.landmark[t].x - wrist.x)**2 + (hand_lms.landmark[t].y - wrist.y)**2) for t in tips]) / 4

                if avg_dist < FIST_THRESHOLD:
                    if arm.is_tracking:
                        arm.reset_to_initial()
                        last_wrist_pos = None
                    cv2.putText(frame, "RESET PAR POING", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Mouvement
                elif arm.is_tracking:
                    target_1 = np.interp(wrist.x, [0, 1], [3000, 1000])
                    target_2 = np.interp(wrist.y, [0, 1], [1500, 3500])

                    if last_wrist_pos is None:
                        arm.current_pos[1], arm.current_pos[2] = target_1, target_2
                        last_wrist_pos = (target_1, target_2)

                    t_tip = hand_lms.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    i_tip = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pinched = np.sqrt((t_tip.x - i_tip.x)**2 + (t_tip.y - i_tip.y)**2) < PINCH_THRESHOLD
                    arm.update_active_dofs(target_1, target_2, pinched)
                
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Output frame - stream to stdout or display locally
        if STREAM_MODE:
            stream_frame(frame)
        else:
            cv2.imshow("Interface SO-101 MediaPipe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                arm.reset_to_initial()
                last_wrist_pos = None
            elif key == ord('b'):
                if not arm.is_tracking and results.multi_hand_landmarks:
                    w_check = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
                    if (BOX_X_MIN < w_check.x < BOX_X_MAX) and (BOX_Y_MIN < w_check.y < BOX_Y_MAX):
                        arm.is_tracking = True
                else:
                    arm.is_tracking = False
                    last_wrist_pos = None

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

if __name__ == "__main__":
    main()