"""MediaPipe capture loop that shares a workspace with Unity OSC packets.

This module replaces the legacy WebSocket broadcaster with a small "fusion
workspace" that receives two asynchronous feeds:

1. Upper-body data from Unity over OSC (Quest HMD, torso, and arm joints).
2. Lower-body landmarks from MediaPipe Pose running on a standard webcam.

Both feeds are logged from the same process so that future work can combine
(lower-body, MediaPipe) + (upper-body, Quest) into a single, fused skeleton.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

import cv2
import mediapipe as mp

try:
    from .osc_pose_receiver import run_server as start_osc_server
except ImportError:
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from pose_stream_server.osc_pose_receiver import run_server as start_osc_server  # type: ignore

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

POSE_LANDMARKS = list(mp_pose.PoseLandmark)


@dataclass
class MediaPipeSnapshot:
    timestamp: float
    landmarks: Mapping[str, Mapping[str, float]]


class FusionWorkspace:
    """Central rendezvous point for Unity OSC packets and MediaPipe landmarks."""

    def __init__(self) -> None:
        self.latest_upper_body: Optional[Mapping[str, object]] = None
        self.latest_mediapipe: Optional[MediaPipeSnapshot] = None

    # ------------------------------------------------------------------
    # Unity / OSC callbacks
    # ------------------------------------------------------------------
    def handle_quest_packet(self, packet: Mapping[str, object], addr) -> None:
        self.latest_upper_body = packet
        timestamp = packet.get("timestamp")
        joint_count = len(packet.get("joints", []) or [])
        logger.info(
            "Unity OSC packet from %s @ %.3f with %d joints", addr, timestamp, joint_count
        )
        self._log_workspace_state()

    # ------------------------------------------------------------------
    # MediaPipe callbacks
    # ------------------------------------------------------------------
    def update_pose(self, snapshot: MediaPipeSnapshot) -> None:
        self.latest_mediapipe = snapshot
        hips = snapshot.landmarks.get("left_hip") or snapshot.landmarks.get("right_hip")
        if hips:
            logger.info(
                "MediaPipe pose @ %.3f hip=(%.3f, %.3f, %.3f) (%d landmarks)",
                snapshot.timestamp,
                hips.get("x", 0.0),
                hips.get("y", 0.0),
                hips.get("z", 0.0),
                len(snapshot.landmarks),
            )
        else:
            logger.info(
                "MediaPipe pose @ %.3f (hips not visible, %d landmarks)",
                snapshot.timestamp,
                len(snapshot.landmarks),
            )
        self._log_workspace_state()

    # ------------------------------------------------------------------
    def _log_workspace_state(self) -> None:
        """Log when both streams are live to highlight the fusion rendezvous."""

        if not self.latest_upper_body or not self.latest_mediapipe:
            return

        quest_ts = float(self.latest_upper_body.get("timestamp", 0.0))
        mediapipe_ts = self.latest_mediapipe.timestamp
        delta = mediapipe_ts - quest_ts

        logger.info(
            "Fusion workspace ready (Quest ts=%.3f, MediaPipe ts=%.3f, Δ=%.3fs) — this is the hook for blending the two bodies.",
            quest_ts,
            mediapipe_ts,
            delta,
        )


# ----------------------------------------------------------------------
# MediaPipe helpers
# ----------------------------------------------------------------------

def _landmark_dict(landmark) -> Dict[str, float]:
    return {
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": float(getattr(landmark, "visibility", 0.0)),
    }


def extract_pose_landmarks(results) -> Optional[MediaPipeSnapshot]:
    if not (results.pose_landmarks and results.pose_world_landmarks):
        return None

    world_landmarks = results.pose_world_landmarks.landmark
    output: MutableMapping[str, Mapping[str, float]] = {}
    for landmark_enum in POSE_LANDMARKS:
        idx = landmark_enum.value
        output[landmark_enum.name.lower()] = _landmark_dict(world_landmarks[idx])

    return MediaPipeSnapshot(timestamp=time.time(), landmarks=dict(output))


async def mediapipe_loop(camera_index: int, workspace: FusionWorkspace) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera index %s", camera_index)
        return

    try:
        while True:
            success, image = cap.read()
            if not success:
                logger.warning("Empty frame, retrying...")
                await asyncio.sleep(0.1)
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            snapshot = extract_pose_landmarks(results)
            if snapshot:
                workspace.update_pose(snapshot)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            cv2.imshow("MediaPipe Pose", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                logger.info("ESC pressed, stopping MediaPipe loop")
                break

            await asyncio.sleep(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--osc-host", default="0.0.0.0", help="Interface for Unity OSC packets (default: 0.0.0.0)")
    parser.add_argument("--osc-port", type=int, default=9000, help="UDP port for Unity OSC packets (default: 9000)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    workspace = FusionWorkspace()
    osc_task = asyncio.create_task(
        start_osc_server(args.osc_host, args.osc_port, workspace.handle_quest_packet)
    )

    try:
        await mediapipe_loop(args.camera_index, workspace)
    finally:
        osc_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await osc_task


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Shutting down MediaPipe + OSC fusion workspace")


if __name__ == "__main__":
    main()
