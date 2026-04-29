# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-17

### Added

- **YOLO-Pose GPU Tracking**
  - New orange "YOLO-Pose GPU" button in the Play page
  - GPU-accelerated pose detection using YOLO11n-pose model
  - Real-time body skeleton overlay on camera feed
  - Expected performance: 25-30 FPS on Jetson Orin Nano

- **Dynamic FPS Counter**
  - Real-time FPS display during tracking
  - Shows detection status (detected/not detected)
  - Displays pan/tilt angles

- **Jetson GPU Setup Script**
  - New `gpu_setup_jetson.sh` script for easy installation
  - Automatic JetPack version detection
  - Correct PyTorch NVIDIA wheels for Jetson

### Changed

- **Camera Feed Selection**
  - Tracking now uses correct video feed (tracker output with skeleton)
  - Fixed incorrect camera switching between 3D and camera views
  - Smooth transition without requiring multiple toggle attempts

- **Video Feed Logic**
  - Tracking activation order fixed (currentTracking set before view change)
  - Proper video source switching based on active tracking mode

### Fixed

- **Tracking Not Displaying**
  - Fixed pose_yolo mode not being handled in main run loop
  - Added proper robot control for YOLO-Pose detection results

- **3D Robot Model Errors**
  - Fixed "getEndEffectorPosition is not a function" error
  - Updated Three.js rotation extraction method for compatibility

- **Video Load Error**
  - Removed invalid video.load() call that caused console error

- **MediaPipe Pose Detector**
  - Fixed AttributeError when MediaPipe not available
  - Added proper initialization of _mp_pose_detector

### Removed

- None

### Technical

- Added YOLO-Pose GPU acceleration support
- Implemented TensorRT-compatible YOLO inference
- Updated Three.js matrix operations for newer versions
- Added proper MJPEG streaming for tracker camera feed

---

## [0.9.0] - 2026-04-16

### Added

- Initial PARC Remote Lab release
- Play page with robot arm control
- 3D visualization with Three.js
- Camera-based gesture/body tracking
- MediaPipe integration for face, hand, and pose detection
- YOLO object detection for gripper camera
- Flask web interface with multiple pages
- Robot arm control with PID tracking

[Unreleased]: https://github.com/your-repo/PARC-Remote-Lab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-repo/PARC-Remote-Lab/releases/tag/v1.0.0
[0.9.0]: https://github.com/your-repo/PARC-Remote-Lab/releases/tag/v0.9.0
