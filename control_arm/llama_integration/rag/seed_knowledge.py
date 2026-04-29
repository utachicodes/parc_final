"""
Seed script to populate RAG knowledge base with PARC Robotics documentation.
Run this once to initialize the knowledge base.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def seed_kinematics_docs(rag_system):
    """Seed kinematics documentation."""
    logger.info("Seeding kinematics documentation...")

    docs = [
        {
            "text": """# Forward Kinematics

Forward kinematics calculates the position and orientation of the robot's end-effector
given the joint angles. For a 6-DOF robot arm like SO-101:

## Formula

The DH parameters define the transformation between joints:
- theta_i: joint angle
- d_i: link offset
- a_i: link length
- alpha_i: link twist

The transformation matrix from base to end-effector is:
T = A1 * A2 * A3 * A4 * A5 * A6

Where each Ai is computed using:
Ai = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

## Code Example

```python
import numpy as np

def forward_kinematics(joint_angles):
    '''
    Compute end-effector position from joint angles.
    joint_angles: [pan, lift, elbow, wrist_flex, wrist_roll, gripper]
    Returns: [x, y, z, roll, pitch, yaw]
    '''
    # DH parameters for SO-101 (example values)
    dh_params = [
        {'theta': joint_angles[0], 'd': 50, 'a': 0, 'alpha': 90},      # Base rotation
        {'theta': joint_angles[1], 'd': 0, 'a': 100, 'alpha': 0},     # Shoulder
        {'theta': joint_angles[2], 'd': 0, 'a': 100, 'alpha': 0},       # Elbow
        {'theta': joint_angles[3], 'd': 0, 'a': 80, 'alpha': 0},       # Wrist
        {'theta': joint_angles[4], 'd': 0, 'a': 50, 'alpha': 0},       # Wrist roll
    ]

    # Compute cumulative transformation
    T = np.eye(4)
    for params in dh_params:
        T = T @ dh_to_transform(**params)

    return T[:3, 3], T[:3, :3]
```""",
            "source": "kinematics_forward.md",
            "category": "kinematics"
        },
        {
            "text": """# Inverse Kinematics

Inverse kinematics (IK) solves for joint angles given a desired end-effector position.
This is more complex than forward kinematics as multiple solutions may exist.

## Approaches

1. **Analytical IK**: Closed-form solutions for specific robot geometries
2. **Numerical IK**: Iterative optimization (CCD, Jacobian Inverse)
3. **Heuristic IK**: Learning-based approaches

## Code Example (Numerical IK using Jacobian)

```python
import numpy as np

def inverse_kinematics_numerical(
    target_pos,
    target_orient,
    initial_angles,
    max_iterations=100,
    tolerance=0.01
):
    '''
    Numerical IK using Jacobian Transpose method.
    '''
    joint_angles = np.array(initial_angles, dtype=float)
    learning_rate = 0.1

    for _ in range(max_iterations):
        # Current end-effector position
        current_pos, current_orient = forward_kinematics(joint_angles)

        # Position error
        pos_error = target_pos - current_pos

        # Check convergence
        if np.linalg.norm(pos_error) < tolerance:
            break

        # Simplified Jacobian (3x6 for position only)
        J = compute_jacobian(joint_angles)

        # Update joint angles
        delta_q = learning_rate * np.linalg.pinv(J) @ pos_error
        joint_angles += delta_q

        # Apply joint limits
        joint_angles = np.clip(joint_angles, -np.pi, np.pi)

    return joint_angles.tolist()
```""",
            "source": "kinematics_inverse.md",
            "category": "kinematics"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_programming_docs(rag_system):
    """Seed programming documentation."""
    logger.info("Seeding programming documentation...")

    docs = [
        {
            "text": """# SCServo SDK for Python

The SO-101 robot arm uses SCServo motors controlled via the SCServo SDK.

## Installation

```bash
pip install scservo-sdk
```

## Basic Usage

```python
from scservo_sdk import PortHandler, sms_sts

# Initialize port
port = PortHandler('/dev/ttyUSB0')
port.openPort()
port.setBaudRate(1000000)

# Create servo controller
servo = sms_sts(port)

# Read current position (returns 0-4095, center=2048)
position = servo.ReadPos(1)  # Servo ID 1
print(f"Position: {position}")

# Move to position (0-4095)
# 2048 = center, 4095 = max one direction, 0 = max other
servo.WritePos(1, 2048, 1000, 50)  # ID, position, speed, acceleration

# Close port when done
port.closePort()
```

## Servo IDs

- ID 1: Shoulder Pan (base rotation)
- ID 2: Shoulder Lift
- ID 3: Elbow Flex
- ID 4: Wrist Flex
- ID 5: Wrist Roll
- ID 6: Gripper""",
            "source": "scservo_basics.py",
            "category": "programming"
        },
        {
            "text": """# Robot Modes and Presets

The SO-101 robot has predefined modes for common operations.

## Available Modes

### Idle Mode
Safe resting position with slight joint bends.
```python
idle_angles = {
    'shoulder_pan': 0,
    'shoulder_lift': 10,
    'elbow_flex': -10,
    'wrist_flex': 0,
    'wrist_roll': 0,
    'gripper': 0
}
```

### Hello Wave
Friendly greeting gesture.
```python
hello_angles = {
    'shoulder_pan': 25,
    'shoulder_lift': -30,
    'elbow_flex': 50,
    'wrist_flex': 10,
    'wrist_roll': 50,
    'gripper': -20
}
```

### Pick and Place
```python
pick_angles = {
    'shoulder_pan': -35,
    'shoulder_lift': -20,
    'elbow_flex': 30,
    'wrist_flex': 20,
    'wrist_roll': 0,
    'gripper': 40  # Closed for pick
}

place_angles = {
    'shoulder_pan': 35,
    'shoulder_lift': -10,
    'elbow_flex': 40,
    'wrist_flex': 15,
    'wrist_roll': 0,
    'gripper': -40  # Open for place
}
```

### Dance Mode
```python
dance_angles = {
    'shoulder_pan': 0,
    'shoulder_lift': -40,
    'elbow_flex': 55,
    'wrist_flex': 15,
    'wrist_roll': 0,
    'gripper': -30
}
```""",
            "source": "robot_modes.md",
            "category": "programming"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_calibration_docs(rag_system):
    """Seed calibration documentation."""
    logger.info("Seeding calibration documentation...")

    docs = [
        {
            "text": """# Servo Calibration Guide

Proper calibration ensures accurate robot positioning.

## Why Calibrate?

- Motor tolerances cause position variations
- Gear backlash affects repeatability
- Center positions may not be exactly at 2048

## Calibration Steps

1. **Power off the robot** and free all joints
2. **Manually move** each joint to its center position
3. **Read the encoder values** at center
4. **Store calibration offsets** in the knowledge base

## Calibration Code

```python
from scservo_sdk import PortHandler, sms_sts

def calibrate_servo(port, servo_id):
    '''
    Calibrate a single servo to find true center.
    Returns the offset from 2048.
    '''
    servo = sms_sts(port)

    # Move to approximate center
    servo.WritePos(servo_id, 2048, 500, 50)
    time.sleep(1)

    # Read position
    center_reading = servo.ReadPos(servo_id)

    # Fine-tune: sweep and find edges
    # ... (sweep logic)

    # Calculate offset
    offset = center_reading - 2048
    return offset

def save_calibration(offsets, filename='calibration.json'):
    '''Save calibration offsets to file.'''
    import json
    with open(filename, 'w') as f:
        json.dump(offsets, f)

def load_calibration(port, filename='calibration.json'):
    '''Load and apply calibration offsets.'''
    import json
    with open(filename) as f:
        offsets = json.load(f)

    servo = sms_sts(port)
    for servo_id, offset in offsets.items():
        # Apply offset to center position
        calibrated_center = 2048 + offset
        servo.WritePos(int(servo_id), calibrated_center, 0, 0)
```""",
            "source": "calibration_guide.md",
            "category": "calibration"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_concepts_docs(rag_system):
    """Seed robotics concepts documentation."""
    logger.info("Seeding robotics concepts...")

    docs = [
        {
            "text": """# SO-101 Robot Arm Anatomy

The PARC Robotics SO-101 is a 6-DOF (Degrees of Freedom) robot arm.

## Joint Structure

1. **Base/Pan Joint**: Rotates entire arm horizontally (shoulder_pan)
2. **Shoulder Joint**: Raises/lowers the main arm (shoulder_lift)
3. **Elbow Joint**: Bends the forearm (elbow_flex)
4. **Wrist Flex**: Tilts the wrist up/down (wrist_flex)
5. **Wrist Roll**: Rotates wrist horizontally (wrist_roll)
6. **Gripper**: Opens/closes the end-effector (gripper)

## Coordinate System

- X-axis: Forward/backward from robot base
- Y-axis: Left/right (perpendicular to X)
- Z-axis: Up/down (vertical)

## Workspace

Maximum reach: approximately 350mm from base center
Gripper span: 0-60mm""",
            "source": "robot_anatomy.md",
            "category": "concepts"
        },
        {
            "text": """# Vision Tracking Modes

The SO-101 supports multiple vision-based tracking modes.

## Face Tracking
Tracks human face position. Robot tilts to follow the face.
- Uses MediaPipe Face Detection
- Suitable for human-robot interaction

## Hand Tracking
Tracks hand position and gestures.
- Uses MediaPipe Hands
- Good for pick-and-place by demonstration
- Gesture recognition for commands

## Pose Tracking
Tracks body pose landmarks (full skeleton).
- Uses MediaPipe Pose
- Maps body movement to robot movement
- Good for teleoperation

## Body Tracking (Full)
Advanced tracking with shoulder, elbow, wrist mapping.
- Maps arm joints directly to human arm
- Natural teleoperation control
- Requires good camera angle""",
            "source": "vision_tracking.md",
            "category": "vision"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_all():
    """Seed the complete knowledge base."""
    import os
    import sys

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.makedirs(os.path.expanduser("~/models/rag"), exist_ok=True)

    from llama_integration.rag.rag_system import RAGSystem

    rag = RAGSystem()

    # Check if already seeded
    if rag.count_documents() > 0:
        logger.info(f"Knowledge base already has {rag.count_documents()} documents")
        response = input("Reset and reseed? (y/N): ")
        if response.lower() != 'y':
            logger.info("Seeding cancelled")
            return
        rag.reset()

    logger.info("Seeding PARC Robotics knowledge base...")

    seed_kinematics_docs(rag)
    seed_programming_docs(rag)
    seed_calibration_docs(rag)
    seed_concepts_docs(rag)

    logger.info(f"Done! Knowledge base has {rag.count_documents()} documents")


if __name__ == "__main__":
    seed_all()
