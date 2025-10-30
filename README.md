# pose-estimation-by-camera-and-vr-headset

## Getting started

This is a hybrid framework for human body pose estimation using **MediaPipe** and external cameras, synchronized with **Meta Quest** tracking data for VR applications.

It allows you to:
- Estimate full-body pose with MediaPipe on a PC server.
- Stream 3D joint data over WebSocket/UDP.
- Receive and apply that pose to a VR avatar in Unity on Meta Quest.
- Fuse headset and MediaPipe data for more accurate tracking.

## System Architecture


## Installation

1. Setup Python env:
Requires Python >=3.9

```
python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run MediaPipe server:
```
python pose_stream_server/mediapipe_stream_server.py
```

