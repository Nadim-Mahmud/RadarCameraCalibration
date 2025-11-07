# mmWave Radar ⇄ Camera Calibration

Repository for calibrating mmWave radar (TI AWR1843) to a monocular camera using the TI demo firmware as the radar data source. The pipeline captures synchronized radar and camera data, extracts correspondences, estimates extrinsics, and provides tools for visualization and validation.

## Features
- Data capture helpers for TI AWR1843 demo output and camera
- Preprocessing and synchronization utilities
- Intrinsic camera calibration utilities (OpenCV-compatible)
- Radar-to-camera extrinsic calibration (correspondence extraction + optimization)
- Visualization and evaluation scripts

## Hardware
- TI AWR1843 mmWave radar (using TI out-of-the-box demo firmware)
- Monocular camera (preferably global shutter)
- Mounting rig with adjustable baseline and orientation
- USB/serial connection to radar and camera host computer

## Software prerequisites
- Python 3.8+ (recommended)
- OpenCV (cv2)
- numpy, scipy
- matplotlib

Install typical Python deps:
```
python -m pip install opencv-python numpy scipy matplotlib pyserial
```

## Calibration pipeline (high level)
1. Capture synchronized radar frames and camera images with timestamps.
2. Preprocess radar data (range-Doppler/angle extraction / point-cloud generation).
3. Undistort images and detect calibration targets / features.
4. Establish radar-to-image correspondences (manual annotation or automated detection).
5. Optimize extrinsic parameters using reprojection/error metrics.
6. Validate using held-out frames and visualization overlays.

## Data format
- Raw captures: data/raw/{session}/
    - camera/: timestamped images (JPEG/PNG)
    - radar/: raw radar frames or TI demo logs
- Processed: data/processed/{session}/
    - pointclouds/*.npy or .pcd
    - timestamps.csv
    - detections/*.json

## Tips
- Use a strong calibration target visible to both modalities (corner reflectors or checkerboard + radar reflectors).
- Keep good temporal synchronization. Record host timestamps for both devices.
- Start with coarse initial extrinsics to improve convergence.
- Visualize reprojection frequently to detect outliers.