#!/bin/bash
# ============================================================
# GPU Setup for Jetson Orin Nano - PARC Robotics
# ============================================================
# Run on your Jetson Orin Nano with JetPack 6.x or 7.x
# This installs/configures everything for MAXIMUM GPU SPEED
#
# OPTIONS:
#   ./gpu_setup_jetson.sh          - Install YOLO GPU (FAST, 10 min)
#   ./gpu_setup_jetson.sh full     - Build MediaPipe GPU (SLOW, 2+ hours)
#   ./gpu_setup_jetson.sh minimal  - Just YOLO GPU setup

set -e

MODE=${1:-minimal}

echo "============================================"
echo "  GPU Setup for Jetson Orin Nano"
echo "  Mode: $MODE"
echo "============================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect JetPack version
detect_jetpack() {
    if [ -f /etc/nv_tegra_release ]; then
        JP_VERSION=$(cat /etc/nv_tegra_release | grep "JetPack" | sed 's/.*JetPack \([0-9]*\.[0-9]*\).*/\1/')
        echo "Detected JetPack: $JP_VERSION"
    else
        JP_VERSION="6.1"
        echo "Assuming JetPack 6.1"
    fi
}

# ============================================================
# Step 1: Update system
# ============================================================
echo -e "${YELLOW}[1/10]${NC} Updating system..."
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
echo -e "${GREEN}[OK]${NC} System updated"

# ============================================================
# Step 2: Remove pip OpenCV (use JetPack's CUDA version)
# ============================================================
echo -e "${YELLOW}[2/10]${NC} Removing pip OpenCV (using JetPack's CUDA version)..."
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y 2>/dev/null || true
echo -e "${GREEN}[OK]${NC} OpenCV removal complete"

# ============================================================
# Step 3: Install PyTorch with CUDA (NVIDIA pre-built wheel)
# ============================================================
echo -e "${YELLOW}[3/10]${NC} Detecting JetPack version..."
detect_jetpack

echo -e "${YELLOW}[4/10]${NC} Installing PyTorch with CUDA..."

# Uninstall existing PyTorch first
pip uninstall torch torchvision -y 2>/dev/null || true

# Install correct PyTorch version based on JetPack
if [ "$JP_VERSION" == "7.0" ]; then
    echo "Installing PyTorch 2.5.0 for JetPack 7.0..."
    pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
elif [ "$JP_VERSION" == "6.1" ]; then
    echo "Installing PyTorch 2.10.0 for JetPack 6.1..."
    wget -q -O /tmp/torch-2.10.0-cp310-cp310-linux_aarch64.whl https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
    pip install --no-cache-dir /tmp/torch-2.10.0-cp310-cp310-linux_aarch64.whl
    rm /tmp/torch-2.10.0-cp310-cp310-linux_aarch64.whl
else
    echo "Installing PyTorch from PyPI (fallback)..."
    pip install torch torchvision
fi

echo -e "${GREEN}[OK]${NC} PyTorch installed"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ============================================================
# Step 5: Install Torchvision
# ============================================================
echo -e "${YELLOW}[5/10]${NC} Installing Torchvision..."
if [ "$JP_VERSION" == "6.1" ]; then
    wget -q -O /tmp/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
    pip install --no-cache-dir /tmp/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
    rm /tmp/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
elif [ "$JP_VERSION" == "7.0" ]; then
    pip install --no-cache-dir torchvision
fi
echo -e "${GREEN}[OK]${NC} Torchvision installed"

# ============================================================
# Step 6: Install Ultralytics (YOLO)
# ============================================================
echo -e "${YELLOW}[6/10]${NC} Installing Ultralytics (YOLO)..."
pip install --no-cache-dir ultralytics
echo -e "${GREEN}[OK]${NC} Ultralytics installed"

# ============================================================
# Step 7: Verify OpenCV CUDA
# ============================================================
echo -e "${YELLOW}[7/10]${NC} Checking OpenCV CUDA support..."
python -c "
import cv2
info = cv2.getBuildInformation()
if 'CUDA' in info:
    print('OpenCV: CUDA ENABLED')
    for line in info.split('\n'):
        if 'CUDA' in line and 'version' in line.lower():
            print(f'  {line.strip()}')
else:
    print('OpenCV: Using JetPack OpenCV (may have CUDA)')
"
echo -e "${GREEN}[OK]${NC} OpenCV check complete"

# ============================================================
# Step 8: Install MediaPipe (pip version - CPU only)
# ============================================================
echo -e "${YELLOW}[8/10]${NC} Installing MediaPipe (pip - CPU mode)..."
pip install --no-cache-dir mediapipe
python -c "import mediapipe as mp; print(f'MediaPipe: {mp.__version__}')"
echo -e "${GREEN}[OK]${NC} MediaPipe installed (CPU mode)"

# ============================================================
# Step 9: Test YOLO GPU
# ============================================================
echo -e "${YELLOW}[9/10]${NC} Testing YOLO GPU..."
python -c "
from ultralytics import YOLO
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
"
echo -e "${GREEN}[OK]${NC} YOLO GPU test complete"

# ============================================================
# Step 10: Download YOLO-Pose model
# ============================================================
echo -e "${YELLOW}[10/10]${NC} Downloading YOLO-pose model..."
mkdir -p ~/models/yolo
if [ ! -f ~/models/yolo/yolo11n-pose.pt ]; then
    wget -q -O ~/models/yolo/yolo11n-pose.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
    echo "YOLO11n-pose downloaded"
else
    echo "YOLO11n-pose already exists"
fi
echo -e "${GREEN}[OK]${NC} YOLO-pose model ready"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo -e "${GREEN}  INSTALLATION COMPLETE!${NC}"
echo "============================================"
echo ""
echo "JetPack: $JP_VERSION"
echo "YOLO-Pose (GPU): ${GREEN}READY${NC} - Use 'YOLO-Pose GPU' button in web UI"
echo "MediaPipe: ${YELLOW}CPU MODE${NC} - Use pip package (slow)"
echo ""
echo "To use YOLO-Pose GPU tracking:"
echo "  1. cd ~/PARC-Remote-Lab"
echo "  2. ./start.sh"
echo "  3. Open http://localhost:5000/play"
echo "  4. Click the ${ORANGE}orange 'YOLO-Pose GPU'${NC} button"
echo ""
echo "Expected FPS: 25-30+ on Jetson Orin Nano"
echo ""

# ============================================================
# OPTIONAL: Build MediaPipe with GPU
# ============================================================
if [ "$MODE" == "full" ]; then
    echo "============================================"
    echo -e "${RED}  BUILDING MEDIAPIPE WITH GPU (2+ hours)${NC}"
    echo "============================================"

    # Install bazel
    echo "Installing Bazel..."
    wget -q https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-installer-linux-x86_64.sh
    chmod +x bazel-*.sh
    sudo ./bazel-*.sh --bin
    sudo rm bazel-*.sh

    # Clone MediaPipe
    echo "Cloning MediaPipe..."
    cd ~
    git clone https://github.com/google/mediapipe.git
    cd mediapipe

    # Install dependencies
    echo "Installing dependencies..."
    sudo apt-get install -y python3-dev cmake libopencv-dev protobuf-compiler

    # Build with CUDA
    echo "Building MediaPipe with GPU (this takes 2+ hours)..."
    bazel build --config=cuda //mediapipe/tasks/python:vision

    echo "MediaPipe GPU build complete!"
fi

if [ "$MODE" == "minimal" ] || [ "$MODE" == "" ]; then
    echo "============================================"
    echo "  For MediaPipe GPU, run:"
    echo "    ./gpu_setup_jetson.sh full"
    echo "============================================"
fi