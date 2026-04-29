#!/usr/bin/env python3
"""
Quick GPU Test for Jetson Orin Nano - PARC Robotics
Run this to verify GPU acceleration is working
"""
import sys
import time
import numpy as np

def print_status(name, ok, detail=""):
    status = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
    print(f"{status} {name}")
    if detail:
        print(f"   {detail}")

def main():
    print("=" * 50)
    print("  GPU Test for Jetson Orin Nano")
    print("=" * 50)
    print()

    # Test 1: PyTorch CUDA
    print("[1/6] Testing PyTorch CUDA...")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print_status("PyTorch", cuda_ok, f"version {torch.__version__}")
        if cuda_ok:
            print(f"       GPU: {torch.cuda.get_device_name(0)}")
            print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("       CUDA not available - GPU not being used!")
    except ImportError:
        print_status("PyTorch", False, "Not installed")
        cuda_ok = False

    # Test 2: OpenCV CUDA
    print()
    print("[2/6] Testing OpenCV CUDA...")
    try:
        import cv2
        build_info = cv2.getBuildInformation()
        opencv_cuda = "CUDA" in build_info
        print_status("OpenCV CUDA", opencv_cuda, cv2.__version__)
        if not opencv_cuda:
            print("       Using CPU OpenCV (slower)")
    except ImportError:
        print_status("OpenCV", False, "Not installed")

    # Test 3: YOLO GPU inference
    print()
    print("[3/6] Testing YOLO GPU inference...")
    try:
        from ultralytics import YOLO
        import torch

        # Load model
        model = YOLO('yolo11n.pt')
        if cuda_ok:
            model.to('cuda')

        # Benchmark
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        times = []

        for i in range(5):
            t0 = time.time()
            _ = model.predict(dummy, verbose=False)
            if cuda_ok:
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        yolo_ok = cuda_ok
        device = "GPU" if cuda_ok else "CPU"
        print_status("YOLO", yolo_ok, f"{device} mode - {fps:.1f} FPS")
    except Exception as e:
        print_status("YOLO", False, str(e))
        yolo_ok = False

    # Test 4: YOLO-Pose GPU inference
    print()
    print("[4/6] Testing YOLO-Pose GPU inference...")
    try:
        from ultralytics import YOLO
        import torch

        model = YOLO('yolo11n-pose.pt')
        if cuda_ok:
            model.to('cuda')

        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        times = []

        for i in range(5):
            t0 = time.time()
            _ = model.predict(dummy, verbose=False, pose=True)
            if cuda_ok:
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        print_status("YOLO-Pose", cuda_ok, f"{'GPU' if cuda_ok else 'CPU'} mode - {fps:.1f} FPS")
    except Exception as e:
        print_status("YOLO-Pose", False, str(e))

    # Test 5: MediaPipe
    print()
    print("[5/6] Testing MediaPipe...")
    try:
        import mediapipe as mp
        print_status("MediaPipe", True, f"version {mp.__version__}")

        # Try to use GPU delegate
        try:
            from mediapipe.tasks.python.vision import FaceLandmarker
            print("       FaceLandmarker: Available")
        except:
            pass

        print("       Note: MediaPipe pip uses CPU on ARM64")
        print("       For GPU, build from source (see gpu_setup_jetson.sh full)")
    except ImportError:
        print_status("MediaPipe", False, "Not installed")

    # Test 6: Camera access
    print()
    print("[6/6] Testing Camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print_status("Camera", True, f"{w}x{h}")
            else:
                print_status("Camera", False, "Cannot read frame")
            cap.release()
        else:
            print_status("Camera", False, "Cannot open camera")
    except Exception as e:
        print_status("Camera", False, str(e))

    # Summary
    print()
    print("=" * 50)
    print("  Summary")
    print("=" * 50)

    if cuda_ok:
        print("\033[92m✓ GPU is working!\033[0m")
        print("   Use YOLO-Pose GPU tracking in web UI for fastest results")
    else:
        print("\033[91m✗ GPU not detected\033[0m")
        print("   Check JetPack installation")
        print("   Run: nvidia-smi")
        print("   Run: python -c 'import torch; print(torch.cuda.is_available())'")

    print()
    print("For best performance on Jetson:")
    print("  1. Use YOLO-Pose GPU (already integrated)")
    print("  2. Use orange 'YOLO-Pose GPU' button in /play page")
    print("  3. Expect 25-30+ FPS")
    print()

if __name__ == '__main__':
    main()
