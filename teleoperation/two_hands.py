import cv2
import mediapipe as mp
import math
import numpy as np

# ── 1. CONFIGURATION ──
import os
PORT = os.environ.get('TELEOP_PORT', 'COM6')
BAUD = int(os.environ.get('TELEOP_BAUD', '1000000'))
L1, L2 = 150.0, 150.0
SERVO_IDS = {"pan": 1, "tilt": 2, "elbow": 3, "flex": 4, "rot": 5, "grip": 6}
LIMITS = {
    "pan": (1024, 3072), "tilt": (1024, 3072), "elbow": (1024, 3072),
    "flex": (1024, 3072), "rot": (1024, 3072), "grip": (1800, 3000)
}

# ── 2. IK ENGINE (Used for Shoulder/Tilt) ──
def calculate_ik(target_x, target_y):
    dist_sq = target_x**2 + target_y**2
    dist = math.sqrt(dist_sq)
    if dist > (L1 + L2):
        ratio = (L1 + L2) / dist
        target_x *= ratio; target_y *= ratio
        dist_sq = target_x**2 + target_y**2
    
    # Shoulder angle (alpha + beta)
    alpha = math.atan2(target_y, target_x)
    beta = math.acos(np.clip((L1**2 + dist_sq - L2**2) / (2 * L1 * dist), -1, 1))
    return math.degrees(alpha + beta)

# ── 3. ROBOT CLASS ──
class SO101Robot:
    def __init__(self, port_name, baud_rate):
        self.current_vals = {k: 2048 for k in SERVO_IDS}
        self.alpha = 0.15  # Increased smoothing for thumb gestures
        self.connected = False
        try:
            from scservo_sdk import PortHandler, PacketHandler
            self.port = PortHandler(port_name); self.packet = PacketHandler(0)
            if self.port.openPort() and self.port.setBaudRate(baud_rate):
                self.connected = True
        except: print("Robot not connected - Simulation Mode")

    def update_and_send(self, key, target):
        lo, hi = LIMITS[key]
        target = np.clip(target, lo, hi)
        smooth_val = (self.alpha * target) + (1 - self.alpha) * self.current_vals[key]
        self.current_vals[key] = smooth_val
        if self.connected:
            self.packet.write2ByteTxRx(self.port, SERVO_IDS[key], 42, int(smooth_val))
        return int(smooth_val)

# ── 4. MAIN EXECUTION ──
def main():
    robot = SO101Robot(PORT, BAUD)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8) as hands, \
         mp_pose.Pose(min_detection_confidence=0.8) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)

            # --- HUD Overlay ---
            cv2.rectangle(frame, (10, 10), (350, 150), (0, 0, 0), -1)
            cv2.putText(frame, "RIGHT: Pan/Tilt + Thumb Elbow", (20, 40), 1, 1, (0, 255, 0), 1)
            cv2.putText(frame, "LEFT: Wrist + Gripper", (20, 70), 1, 1, (255, 100, 0), 1)

            targets = {}

            if hand_res.multi_hand_landmarks:
                for hand_lms, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                    label = handedness.classification[0].label
                    lm = hand_lms.landmark
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    # --- RIGHT HAND: ARM CONTROL ---
                    if label == "Right" and pose_res.pose_landmarks:
                        p_lm = pose_res.pose_landmarks.landmark
                        
                        # Pan: Left/Right movement of hand
                        targets["pan"] = np.interp(lm[0].x, [0.3, 0.7], [LIMITS["pan"][0], LIMITS["pan"][1]])
                        
                        # Tilt: IK based on hand height relative to shoulder
                        rx = (lm[0].x - p_lm[12].x) * 450
                        ry = (p_lm[12].y - lm[0].y) * 450
                        shoulder_angle = calculate_ik(rx, ry)
                        targets["tilt"] = np.interp(shoulder_angle, [0, 160], [LIMITS["tilt"][0], LIMITS["tilt"][1]])

                        # ELBOW (SERVO 3): THUMB CURVATURE
                        # Measure distance between thumb tip (4) and pinky base (17)
                        t_dist = math.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y)
                        # Mapping: Tucked thumb (0.05) to Outstretched (0.18)
                        targets["elbow"] = np.interp(t_dist, [0.06, 0.18], [LIMITS["elbow"][0], LIMITS["elbow"][1]])
                        
                        cv2.putText(frame, f"Thumb Flex: {t_dist:.2f}", (20, 100), 1, 1, (0, 255, 0), 1)

                    # --- LEFT HAND: WRIST & GRIP ---
                    if label == "Left":
                        targets["rot"] = np.interp(lm[0].x, [0.2, 0.5], [LIMITS["rot"][0], LIMITS["rot"][1]])
                        targets["flex"] = np.interp(lm[0].y, [0.3, 0.7], [LIMITS["flex"][1], LIMITS["flex"][0]])
                        
                        # Gripper: Pinch distance between thumb (4) and index (8)
                        g_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                        targets["grip"] = np.interp(g_dist, [0.02, 0.12], [LIMITS["grip"][1], LIMITS["grip"][0]])

                # Send all gathered targets to servos
                for key, val in targets.items():
                    robot.update_and_send(key, val)

            cv2.imshow('SO101 Hybrid Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()