"""
YOLO-Pose based tracker for PARC Robotics
This replaces MediaPipe pose tracking with YOLO-pose for GPU acceleration on Jetson

YOLO-pose advantages on Jetson:
- Uses PyTorch CUDA (GPU accelerated)
- TensorRT optimization available
- Much faster than MediaPipe on ARM64
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, List, Tuple


class YOLOPoseTracker:
    """
    Pose tracking using YOLO-pose model.
    GPU accelerated via PyTorch CUDA.
    """

    def __init__(self, model_name='yolo11n-pose.pt', conf_threshold=0.5, device='cuda'):
        """
        Initialize YOLO-pose tracker.

        Args:
            model_name: YOLO pose model name (yolo11n-pose.pt, yolo11s-pose.pt, etc.)
            conf_threshold: Confidence threshold for keypoint detection
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.model_name = model_name
        self.model = None
        self._ts = 0
        self._fps = 0.0
        self._frame_times = []
        self._load_model()

    def _load_model(self):
        """Load YOLO-pose model with GPU support."""
        try:
            from ultralytics import YOLO
            import torch

            # Try to load model, download if not found
            try:
                self.model = YOLO(self.model_name)
            except:
                print(f"Downloading {self.model_name}...")
                self.model = YOLO(f'{self.model_name.replace(".pt", "")}')

            # Move to GPU if available
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model.to('cuda')
                print(f"[YOLOPoseTracker] Model loaded with GPU acceleration")
                print(f"[YOLOPoseTracker] GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"[YOLOPoseTracker] Model loaded (CPU mode)")

            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, pose=True)

        except Exception as e:
            print(f"[YOLOPoseTracker] Failed to load model: {e}")
            self.model = None

    def _next_ts(self) -> int:
        self._ts += 1
        return self._ts

    def _update_fps(self):
        """Update FPS calculation."""
        now = time.time()
        self._frame_times.append(now)
        # Keep last 30 frames
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            self._fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])

    def detect_pose(self, bgr_frame) -> Optional[Dict[str, Any]]:
        """
        Detect pose in frame.

        Returns:
            Dict with keys:
                - nose, left_shoulder, right_shoulder, left_elbow, right_elbow,
                  left_wrist, right_wrist, left_hip, right_hip (landmark objects)
                - landmarks: list of all landmarks
                - confidence: detection confidence
        """
        if self.model is None:
            return None

        try:
            # Run inference
            results = self.model.predict(
                bgr_frame,
                conf=self.conf_threshold,
                verbose=False,
                pose=True,
                device=self.device
            )

            self._update_fps()

            if not results or len(results) == 0:
                return None

            r = results[0]
            if r.keypoints is None or len(r.keypoints) == 0:
                return None

            kpts = r.keypoints.data[0]  # Get first person's keypoints

            # COCO keypoint order for YOLO-pose:
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
            # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
            # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

            # Map to our format
            landmarks = []
            for i, kp in enumerate(kpts):
                x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                landmarks.append(type('obj', (object,), {'x': x, 'y': y, 'z': 0, 'confidence': conf}))

            result = {
                'nose': landmarks[0] if len(landmarks) > 0 else None,
                'left_shoulder': landmarks[5] if len(landmarks) > 5 else None,
                'right_shoulder': landmarks[6] if len(landmarks) > 6 else None,
                'left_elbow': landmarks[7] if len(landmarks) > 7 else None,
                'right_elbow': landmarks[8] if len(landmarks) > 8 else None,
                'left_wrist': landmarks[9] if len(landmarks) > 9 else None,
                'right_wrist': landmarks[10] if len(landmarks) > 10 else None,
                'left_hip': landmarks[11] if len(landmarks) > 11 else None,
                'right_hip': landmarks[12] if len(landmarks) > 12 else None,
                'landmarks': landmarks,
                'confidence': r.probs.top1conf if r.probs else 1.0,
                'fps': self._fps
            }

            return result

        except Exception as e:
            print(f"[YOLOPoseTracker] Detection error: {e}")
            return None

    def draw_pose(self, frame, result, show_confidence=True):
        """Draw pose skeleton on frame."""
        if result is None or result['landmarks'] is None:
            return frame

        h, w = frame.shape[:2]
        landmarks = result['landmarks']

        # Define skeleton connections (COCO format)
        skeleton = [
            (5, 6),   # shoulders
            (5, 7),   # left arm
            (7, 9),   # left forearm
            (6, 8),   # right arm
            (8, 10),  # right forearm
            (5, 11),  # left torso
            (6, 12),  # right torso
            (11, 12), # hips
            (11, 13), # left thigh
            (13, 15), # left shin
            (12, 14), # right thigh
            (14, 16), # right shin
        ]

        # Draw connections
        for i, j in skeleton:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = landmarks[i]
                pt2 = landmarks[j]
                if pt1.confidence > 0.3 and pt2.confidence > 0.3:
                    x1, y1 = int(pt1.x), int(pt1.y)
                    x2, y2 = int(pt2.x), int(pt2.y)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Draw keypoints
        for i, lm in enumerate(landmarks):
            if lm.confidence > 0.3:
                x, y = int(lm.x), int(lm.y)
                color = (0, 255, 255)  # Yellow for all
                cv2.circle(frame, (x, y), 4, color, -1)

        # Draw FPS
        if show_confidence and result.get('fps'):
            cv2.putText(
                frame,
                f"FPS: {result['fps']:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        return frame


class YOLOHandTracker:
    """
    Hand tracking using YOLO with segmentation.
    Uses YOLO object detection with hand class, or segmentation.
    """

    def __init__(self, model_name='yolo11n.pt', device='cuda'):
        """
        Initialize YOLO hand tracker.

        Note: For hands, we recommend using MediaPipe Hands or
        a dedicated hand keypoint model. This is a fallback using
        object detection or custom hand keypoint model.
        """
        self.device = device
        self.model = None
        self._fps = 0.0
        self._frame_times = []
        self._load_model(model_name)

    def _load_model(self, model_name):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            import torch

            self.model = YOLO(model_name)
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model.to('cuda')
                print(f"[YOLOHTracker] Model loaded with GPU")
        except Exception as e:
            print(f"[YOLOHTracker] Failed to load: {e}")

    def detect_hand(self, bgr_frame) -> Optional[Tuple[float, float, str, List]]:
        """
        Detect hand in frame.
        Returns (center_x, center_y, hand_side, landmarks) or None.
        """
        if self.model is None:
            return None

        try:
            results = self.model.predict(
                bgr_frame,
                conf=0.5,
                verbose=False,
                device=self.device
            )

            # Update FPS
            now = time.time()
            self._frame_times.append(now)
            self._frame_times = self._frame_times[-30:]
            if len(self._frame_times) >= 2:
                self._fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])

            if not results or len(results) == 0:
                return None

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return None

            # Find hand boxes (class 0 is person, need hand class)
            # This is a simplified version - for hands, use MediaPipe or dedicated model
            for box in r.boxes:
                cls = int(box.cls[0])
                # Person class - extract hand region
                if cls == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Return center of person region as approximation
                    cx = ((x1 + x2) / 2) / bgr_frame.shape[1]
                    cy = ((y1 + y2) / 2) / bgr_frame.shape[0]
                    return (cx, cy, "Right", None)

            return None

        except Exception as e:
            print(f"[YOLOHTracker] Detection error: {e}")
            return None


# Model download URLs for YOLO pose
YOLO_POSE_MODELS = {
    'yolo11n-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
    'yolo11s-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
    'yolo11m-pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt',
}


if __name__ == '__main__':
    # Test the tracker
    print("Testing YOLO-Pose Tracker...")

    tracker = YOLOPoseTracker(model_name='yolo11n-pose.pt', device='cuda')

    # Use webcam if available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available, using test image")
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = tracker.detect_pose(test_img)
        print(f"Test result: {result is not None}")
    else:
        print("Camera opened, running detection test...")
        ret, frame = cap.read()
        if ret:
            result = tracker.detect_pose(frame)
            print(f"Detection result: {result is not None}")
            if result:
                print(f"FPS: {result.get('fps', 0):.1f}")
        cap.release()
