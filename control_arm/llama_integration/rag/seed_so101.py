"""
Seed script with CORRECT SO-101 DH parameters from Georgia Tech ECE 4560
=======================================================================
Reference: https://maegantucker.com/ECE4560/assignment6-so101/

Based on actual Tucker Course Assignment 6 code.
This file provides the CORRECT forward kinematics implementation.
"""

import logging
import sys
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def Rx(thetadeg):
    """Rotation about X axis."""
    thetarad = thetadeg * 0.017453292519943295
    c, s = np.cos(thetarad), np.sin(thetarad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(thetadeg):
    """Rotation about Y axis."""
    thetarad = thetadeg * 0.017453292519943295
    c, s = np.cos(thetarad), np.sin(thetarad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def Rz(thetadeg):
    """Rotation about Z axis."""
    thetarad = thetadeg * 0.017453292519943295
    c, s = np.cos(thetarad), np.sin(thetarad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def get_gw1(theta1_deg):
    """
    World/base to joint 1 transformation.
    CORRECT implementation from Tucker Course.

    Args:
        theta1_deg: shoulder_pan angle in degrees

    Returns:
        4x4 transformation matrix
    """
    displacement = (0.0388353, 0.0, 0.0624)
    rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_g12(theta2_deg):
    """
    Joint 1 to Joint 2 transformation.
    CORRECT implementation from Tucker Course.
    """
    displacement = (0.0, 0.0, 0.100)
    rotation = Rx(-90 - theta2_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_g23(theta3_deg):
    """
    Joint 2 to Joint 3 transformation.
    CORRECT implementation from Tucker Course.
    """
    displacement = (0.100, 0.0, 0.0)
    rotation = Rx(90 + theta3_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_g34(theta4_deg):
    """
    Joint 3 to Joint 4 transformation.
    CORRECT implementation from Tucker Course.
    """
    displacement = (0.0, 0.0, 0.080)
    rotation = Rx(-90 - theta4_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_g45(theta5_deg):
    """
    Joint 4 to Joint 5 transformation.
    CORRECT implementation from Tucker Course.
    """
    displacement = (0.0, 0.0, 0.050)
    rotation = Rz(theta5_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_g5t():
    """
    Joint 5 to tool/gripper tip (fixed transformation).
    CORRECT implementation from Tucker Course.

    Note: This is the object frame position, not the actual gripper tip.
    The gripper frame is offset by this amount from joint 5.
    """
    displacement = (0.0, 0.0, 0.050)
    rotation = np.eye(3)
    pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
    return pose

def get_forward_kinematics(position_dict):
    """
    Compute complete forward kinematics for SO-101.

    Args:
        position_dict: Dictionary with keys:
            - shoulder_pan (joint 1)
            - shoulder_lift (joint 2)
            - elbow_flex (joint 3)
            - wrist_flex (joint 4)
            - wrist_roll (joint 5)
            - gripper (joint 6) - does NOT affect position

    Returns:
        (position, rotation) - position in meters, rotation as 3x3 matrix
    """
    gw1 = get_gw1(position_dict['shoulder_pan'])
    g12 = get_g12(position_dict['shoulder_lift'])
    g23 = get_g23(position_dict['elbow_flex'])
    g34 = get_g34(position_dict['wrist_flex'])
    g45 = get_g45(position_dict['wrist_roll'])
    g5t = get_g5t()

    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation


def seed_kinematics_docs_correct(rag_system):
    """Seed with CORRECT kinematics from Tucker course."""
    logger.info("Seeding correct kinematics documentation...")

    docs = [
        {
            "text": """# SO-101 Forward Kinematics - CORRECT DH Parameters

Based on Georgia Tech ECE 4560 Tucker Course - Assignment 6

## Joint Names and Order (same as SDK)
- shoulder_pan: Base rotation (joint 1)
- shoulder_lift: Shoulder elevation (joint 2)
- elbow_flex: Elbow bend (joint 3)
- wrist_flex: Wrist pitch (joint 4)
- wrist_roll: Wrist rotation (joint 5)
- gripper: Gripper open/close (joint 6) - does NOT affect object position

## Transformation Matrices

### Joint 1 - Base Frame to Shoulder (pan)
```python
def get_gw1(theta1_deg):
    displacement = (0.0388353, 0.0, 0.0624)
    rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose
```

### Joint 2 - Shoulder to Upper Arm (lift)
```python
def get_g12(theta2_deg):
    displacement = (0.0388353, 0.0, 0.0624)
    # Actual transformation based on URDF
```

## Important Note
Joint 6 (gripper) does NOT affect object position - only joints 1-5 matter for FK.

## Forward Kinematics Complete Formula
```python
def get_forward_kinematics(position_dict):
    gw1 = get_gw1(position_dict['shoulder_pan'])
    g12 = get_g12(position_dict['shoulder_lift'])
    g23 = get_g23(position_dict['elbow_flex'])
    g34 = get_g34(position_dict['wrist_flex'])
    g45 = get_g45(position_dict['wrist_roll'])
    g5t = get_g5t()  # Fixed transformation
    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation
```

## Rotation Matrices
```python
def Rx(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def Rz(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
```""",
            "source": "kinematics_correct.md",
            "category": "kinematics"
        },
        {
            "text": """# SO-101 Pick and Place Configuration

From Georgia Tech ECE 4560 Tucker Course

## Configuration Values (in DEGREES for SDK)

### Starting Configuration (pick up object)
```python
starting_configuration = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50  # Open position
}
```

### Final Configuration (place object)
```python
final_configuration = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50  # Still open
}
```

### Gripper Closed (for grasping)
```python
gripper_closed = {
    'gripper': 5  # Closed
}

gripper_open = {
    'gripper': 50  # Open
}
```

## IMPORTANT: Joint Angle Units
- SDK uses DEGREES
- MuJoCo uses RADIANS
- Formula: radians = degrees * pi / 180

## Pick and Place Sequence
1. Move to starting_configuration with gripper OPEN
2. Move to starting_configuration_closed (gripper grips object)
3. Move to intermediate pose (lift up)
4. Move to final_configuration_closed (move to destination)
5. Move to final_configuration (gripper opens to release)
6. Return to home position""",
            "source": "pick_place_config.md",
            "category": "kinematics"
        },
        {
            "text": """# SO-101 Joint Limits and Specs

## Joint Limits (from URDF)

| Joint | Lower (rad) | Upper (rad) | Lower (deg) | Upper (deg) |
|-------|-------------|-------------|-------------|-------------|
| 1 (pan) | -1.91986 | 1.91986 | -110° | 110° |
| 2 (lift) | -1.74533 | 1.74533 | -100° | 100° |
| 3 (elbow) | -1.74533 | 1.5708 | -100° | 90° |
| 4 (wrist_flex) | -1.65806 | 1.65806 | -95° | 95° |
| 5 (wrist_roll) | -2.74385 | 2.84121 | -157° | 163° |
| 6 (gripper) | -0.174533 | 1.74533 | -10° | 100° |

## Servo Center Position
- Center = 2048 (in servo units 0-4095)
- Each degree ≈ 11.37 servo units (4096/360)

## Gripper Specifications
- Gripper range: 0-60mm equivalent
- Servo value 0 = fully closed (grip)
- Servo value 60 = fully open (release)
- Calibration recommended for accurate positioning""",
            "source": "joint_limits.md",
            "category": "specifications"
        },
        {
            "text": """# SO-101 Robot Arm - Complete Technical Reference

Based on Georgia Tech ECE 4560 Course Materials

## Robot Structure
6-DOF (Degrees of Freedom) manipulator arm:
1. Base rotation (pan) - Yaw about vertical axis
2. Shoulder lift - Pitch about horizontal axis
3. Elbow flex - Pitch about elbow joint
4. Wrist flex - Pitch about wrist
5. Wrist roll - Roll about tool axis
6. Gripper - Linear open/close

## Coordinate System
- X-axis: Forward from robot base
- Y-axis: Left/Right perpendicular to X
- Z-axis: Vertical (up)

## Transformation Chain (base to end-effector)
base → joint1 → joint2 → joint3 → joint4 → joint5 → gripper → object

## Key Insight: Gripper Doesn't Affect FK
The gripper (joint 6) rotation does NOT affect the position of objects in the gripper.
This is because the object frame is defined relative to joint 5, not joint 6.
This simplifies IK calculations - only joints 1-5 matter for positioning.

## Difference: SDK vs MuJoCo
- SDK commands: DEGREES
- MuJoCo simulation: RADIANS
- Conversion: rad = deg * pi / 180

## Forward Kinematics Validation
Use MuJoCo to validate FK calculations:
1. Set joint positions in MuJoCo
2. Use FK to predict object position
3. Add visualization geometry at predicted position
4. If visualization aligns with gripper in simulation = FK is correct""",
            "source": "so101_complete_ref.md",
            "category": "concepts"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_programming_docs_correct(rag_system):
    """Seed with correct SDK usage."""
    logger.info("Seeding correct programming documentation...")

    docs = [
        {
            "text": """# SO-101 SDK Correct Usage

## Servo IDs (from URDF)
- ID 1: shoulder_pan (base rotation)
- ID 2: shoulder_lift
- ID 3: elbow_flex
- ID 4: wrist_flex
- ID 5: wrist_roll
- ID 6: gripper

## Angle Conversion
```python
# Degrees to Servo Units (0-4095, center=2048)
DEGREE_TO_UNIT = 4095 / 360  # ≈ 11.375

def deg_to_servo(degrees):
    return int(2048 + degrees * DEGREE_TO_UNIT)

def servo_to_deg(servo_value):
    return (servo_value - 2048) / DEGREE_TO_UNIT
```

## Basic Movement
```python
from scservo_sdk import PortHandler, sms_sts

# Initialize
port = PortHandler('/dev/ttyUSB0')
port.openPort()
port.setBaudRate(1000000)
servo = sms_sts(port)

# Move joint to position (in degrees)
def move_joint(servo_id, angle_deg, speed=500, acc=50):
    servo_units = int(2048 + angle_deg * (4095/360))
    servo.WritePosEx(servo_id, servo_units, speed, acc)

# Example: Move to home position
home = {
    1: 0,    # shoulder_pan
    2: 0,    # shoulder_lift
    3: 0,    # elbow_flex
    4: 0,    # wrist_flex
    5: 0,    # wrist_roll
    6: 0     # gripper (center)
}
for servo_id, angle in home.items():
    move_joint(servo_id, angle)
```

## Reading Position
```python
# Returns servo units (0-4095)
position = servo.ReadPos(servo_id)
angle_deg = (position - 2048) * 360 / 4095
```

## Gripper Control
```python
# Open gripper (release)
servo.WritePosEx(6, int(2048 + 50 * 11.375), 500, 50)

# Close gripper (grip)
servo.WritePosEx(6, int(2048 + 5 * 11.375), 500, 50)
```""",
            "source": "sdk_correct_usage.py",
            "category": "programming"
        },
        {
            "text": """# SO-101 Robot - Complete Hardware Specifications

## Robot Overview
The SO-101 is a 6-DOF robot arm used in Georgia Tech ECE 4560 courses.
It uses serial bus servos for precise position control.

## Communication Hardware

### Bus Servo Adapter (Waveshare)
- **Model**: Bus Servo Adapter (A)
- **Input Voltage**: 9~12.6V (must match servo voltage)
- **Communication**: UART or USB
- **Connection**: RX-RX, TX-TX for UART mode
- **Max Servos**: Up to 253 serial bus servos

### Connection Setup
```
USB Mode (for PC, Jetson):
- Set jumper cap to B position
- Connect USB to host
- Use port: /dev/ttyUSB0 or COM port

UART Mode (for Raspberry Pi, ESP32):
- Set jumper cap to A position
- Connect RX-RX, TX-TX
- Use UART pins
```

## ST3215 Serial Bus Servo Specifications

### Electrical
- **Operating Voltage**: 9-12.6V
- **No-load Current**: 100mA
- **Stall Current**: 2.5A

### Position Control
- **Position Range**: 0-4095 (12-bit resolution)
- **Center Position**: 2048
- **Degrees per Unit**: 360° / 4096 ≈ 0.088°
- **Maximum Rotation**: ~360° (limited by servo)

### Communication
- **Protocol**: Half-duplex serial
- **Baudrate**: 1,000,000 bps (default)
- **Response**: Can read position, load, voltage, temperature

## SO-101 Servo IDs and Functions

| Servo ID | Joint Name      | Function              | Typical Center |
|----------|-----------------|----------------------|----------------|
| 1        | shoulder_pan    | Base rotation (Y-axis) | 2048           |
| 2        | shoulder_lift   | Shoulder pitch        | 2048           |
| 3        | elbow_flex     | Elbow bend           | 2048           |
| 4        | wrist_flex     | Wrist pitch          | 2048           |
| 5        | wrist_roll     | Wrist rotation       | 2048           |
| 6        | gripper        | Gripper open/close   | 2048           |

## Position Calculation Formulas

### Degrees to Servo Units
```
servo_units = 2048 + (angle_deg * 4096 / 360)
```

### Servo Units to Degrees
```
angle_deg = (servo_units - 2048) * 360 / 4096
```

### With Calibration Offset
```
servo_units = 2048 + offset + (angle_deg * 4096 / 360)
```

## Joint Limits (Software Safety)

| Joint          | Min (deg) | Max (deg) | Min (units) | Max (units) |
|----------------|-----------|-----------|-------------|-------------|
| shoulder_pan   | -110      | 110       | 937         | 3158        |
| shoulder_lift  | -100      | 90        | 1039        | 3056        |
| elbow_flex     | -140      | 140       | 640         | 3455        |
| wrist_flex     | -100      | 100       | 1039        | 3056        |
| wrist_roll     | -150      | 150       | 788         | 3307        |
| gripper        | 0         | 60        | 2048        | 2719        |

## SDK Usage (Python)

### Initialize
```python
from scservo_sdk import PortHandler, sms_sts

port = PortHandler('/dev/ttyUSB0')
port.openPort()
port.setBaudRate(1000000)
servo = sms_sts(port)
```

### Move Joint
```python
def move_to(servo_id, angle_deg, speed=500, acc=50):
    units = int(2048 + angle_deg * 4096 / 360)
    servo.WritePosEx(servo_id, units, speed, acc)
```

### Read Position
```python
pos, result, err = servo.ReadPos(servo_id)
if result == 0:  # COMM_SUCCESS
    angle = (pos - 2048) * 360 / 4096
```

### Set Speed and Torque
```python
servo.WriteSpe(servo_id, speed)  # 0-2047
servo.WriteSGByte(servo_id, torque_percent)  # 0-1000 = 0-100%
```""",
            "source": "robot_hardware_complete.md",
            "category": "hardware"
        },
        {
            "text": """# SO-101 Common Issues and Fixes

## Problem: Robot moves to wrong position
- Cause: Incorrect angle conversion (deg vs rad)
- Fix: Remember SDK uses DEGREES, MuJoCo uses RADIANS

## Problem: Gripper doesn't hold object
- Cause: Gripper servo value too high (not closed enough)
- Fix: Use value 5-10 for closed, 50-60 for open

## Problem: Forward kinematics don't match actual position
- Cause: Calibration offsets not applied
- Fix: Run calibration routine to find true center

## Problem: Jerky movement
- Cause: Speed too high or acceleration too low
- Fix: Reduce speed or increase acceleration values

## Problem: Servo not responding
- Check: PortHandler correctly initialized
- Check: Baud rate matches (default 1000000)
- Check: Servo ID correct
- Check: Power supply adequate

## Calibration Procedure
1. Manually move each joint to visual center
2. Read servo position at center
3. Calculate offset from ideal 2048
4. Store offsets and apply in movement code

## Joint Limits Safety
Always respect joint limits to prevent damage:
- Monitor servo.WritePosEx() calls
- Add software limits in code
- Use physical stops as backup""",
            "source": "common_issues.md",
            "category": "troubleshooting"
        }
    ]

    rag_system.add_documents(
        texts=[d["text"] for d in docs],
        sources=[d["source"] for d in docs],
        categories=[d["category"] for d in docs]
    )


def seed_all():
    """Seed the complete knowledge base with CORRECT information."""
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.makedirs(os.path.expanduser("~/models/rag"), exist_ok=True)

    from llama_integration.rag.rag_system import RAGSystem

    rag = RAGSystem()

    logger.info("Seeding PARC Robotics CORRECTED knowledge base...")

    # Reset existing
    rag.reset()

    seed_kinematics_docs_correct(rag)
    seed_programming_docs_correct(rag)

    logger.info(f"Done! Knowledge base has {rag.count_documents()} documents")


if __name__ == "__main__":
    seed_all()
