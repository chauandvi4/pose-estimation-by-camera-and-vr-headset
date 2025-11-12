# pose-estimation-by-camera-and-vr-headset

## Getting started

This is a hybrid framework for human body pose estimation using **MediaPipe**, **SynthPose MMPose** and external cameras, synchronized with **Meta Quest** tracking data for VR applications.

It allows you to:
- Estimate full-body pose with MediaPipe on a PC server.
- Stream 3D joint data over WebSocket/UDP.
- Receive and apply that pose to a VR avatar in Unity on Meta Quest.
- Fuse headset and MediaPipe data for more accurate tracking.

## System Architecture


## Installation

1. Setup Python env:
Requires Python >=3.10

```
python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run MediaPipe server:
```
python pose_stream_server/mediapipe_stream_server.py
```

3. (Optional) Run the OSC receiver bridge to capture Unity packets without a headset:
```
python pose_stream_server/osc_pose_receiver.py --host 0.0.0.0 --port 9000
```

   This listener prints the JSON payload emitted by the Unity ``OscPoseSender``
   and ``OscPoseSenderStub`` components. It is useful for validating the
   transport layer before wiring the Quest hardware or MediaPipe fusion logic.

## End-to-end OSC smoke test (Unity ↔ Python)

The following steps let you exercise the OSC transport loop even on a macOS
laptop without a connected Quest. The Unity scene will use the
``OscPoseSenderStub`` component, which emits synthetic head motion so that
packets continue to flow.

1. **Start the Python OSC receiver.**

   ```bash
   python pose_stream_server/osc_pose_receiver.py --host 0.0.0.0 --port 9000
   ```

   Expected result: the terminal prints ``Listening for OSC pose packets on``
   ``udp://0.0.0.0:9000``. No packets appear yet because Unity is not running.

2. **Open the Unity project.**

   * Launch Unity Hub and add the ``unity_client`` folder as a project.
   * Open the project with Unity 2021.3 LTS or newer.

   Expected result: the editor loads the sample scene. There is no play-mode
   output yet.

3. **Configure ``OscPoseSenderStub`` in the scene (only required once).**

   * Create an empty GameObject (e.g. ``OSC Pose Sender``).
   * Attach the ``OscPoseSenderStub`` script located under ``Assets/Scripts``.
   * In the Inspector, set **Remote Host** to ``127.0.0.1`` and **Remote Port**
     to ``9000`` so the component targets the local Python listener.
   * Leave **Simulate Hmd When Unavailable** enabled so the script generates
     fake HMD poses.

   Expected result: the component shows runtime-only fields for the HMD pose
   but stays idle while the scene is stopped.

4. **Enter Play Mode in Unity.**

   * Press the **Play** button in the Unity Editor.

   Expected result: the Game view runs. Because the stub is active, the
   component emits simulated head data at ~60 Hz.

5. **Observe OSC packets on the Python side.**

   The terminal running ``osc_pose_receiver.py`` begins logging packets such as:

   ```text
   INFO:__main__:Packet from ('127.0.0.1', 54321) timestamp=1700000000.123 hmd=(0.001, 1.602, -0.048) yaw=2.1
   ```

   * ``timestamp`` is the Unity wall-clock time (seconds since epoch).
   * ``hmd`` shows the simulated head position in meters.
   * ``yaw`` oscillates as the simulated head rotates left/right.

6. **Stop Play Mode.**

   * Click **Play** again in Unity to exit play mode.

   Expected result: Unity stops sending packets. The Python receiver continues
   running but no new log lines appear until Unity re-enters play mode. Press
   ``Ctrl+C`` in the terminal when you are done to stop the receiver.

At this point you have verified the OSC communication loop between Unity and
Python. Once Quest hardware is available, swap the stub for the production
``OscPoseSender`` script and supply real joint bindings for the avatar
hierarchy without changing the network plumbing.

