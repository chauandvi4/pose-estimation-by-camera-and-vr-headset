"""Realtime SynthPose pose estimation without relying on ``MMPoseInferencer``.

This module provides a small helper that can stream frames from a webcam and run
pose detection with the SynthPose HRNet-48 checkpoint that Stanford released on
Hugging Face.  The original demo in this repository relied on
``MMPoseInferencer`` which is available only in recent MMPose versions.  On
systems that ship with an older release, importing that class raises an error
and the demo cannot run.  The code below solves that problem by constructing the
best available inferencer at runtime:

* If ``MMPoseInferencer`` exists we use it directly (same behaviour as the
  original script).
* Otherwise we fall back to ``Pose2DInferencer`` which is present in older
  versions of MMPose.

Each inferencer is wrapped in a tiny adapter so that the rest of the program can
interact with a unified ``__call__(np.ndarray) -> list[np.ndarray]`` interface.

The script performs the following steps:

1. Download the SynthPose snapshot from Hugging Face (unless already present).
2. Auto-discover the config (.py) and checkpoint (.pth) paths inside the
   snapshot.
3. Open a webcam/video stream with OpenCV.
4. Run pose inference on every frame (optionally skipping frames to save
   compute).
5. Draw the 52 predicted keypoints and display the result in a window.

The file exposes multiple functions to keep the logic clean and testable.  The
``main`` function wires everything together so the module can be launched as a
stand-alone script.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Helper dataclasses / utilities
# ---------------------------------------------------------------------------


@dataclass
class PosePrediction:
    """Container holding keypoints for a single person."""

    keypoints: np.ndarray  # shape (K, 3) with (x, y, score)


class InferenceError(RuntimeError):
    """Raised when we cannot build an inferencer."""


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def ensure_model_snapshot(repo_id: str, model_dir: str) -> Path:
    """Download the Hugging Face snapshot if ``model_dir`` does not exist."""

    path = Path(model_dir)
    if path.exists():
        return path

    path.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading SynthPose assets from {repo_id} ...")
        snapshot_download(repo_id=repo_id, local_dir=str(path), local_dir_use_symlinks=False)
        print("Download complete.")
    except Exception as exc:  # pragma: no cover - network code
        print(f"Warning: snapshot download failed ({exc}).")
        print("If you already have the model locally you can ignore this warning.")
    return path


def discover_config_and_checkpoint(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Search ``model_dir`` for the first config (.py) and checkpoint (.pth)."""

    config_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    for file in model_dir.rglob("*.py"):
        if "inference" in file.name or "synthpose" in file.name:
            config_path = file
            break
    if config_path is None:
        for file in model_dir.rglob("*.py"):
            config_path = file
            break

    for file in model_dir.rglob("*.pth"):
        checkpoint_path = file
        break

    return config_path, checkpoint_path


# ---------------------------------------------------------------------------
# Inferencer adapters
# ---------------------------------------------------------------------------


class BaseInferencer:
    """Small abstraction that hides the differences between API versions."""

    def __call__(self, frame: np.ndarray) -> List[PosePrediction]:  # pragma: no cover - thin wrapper
        raise NotImplementedError


class MMPoseInferencerAdapter(BaseInferencer):
    """Adapter that wraps :class:`mmpose.apis.MMPoseInferencer`."""

    def __init__(self, config: Path, checkpoint: Path, device: str) -> None:
        from mmpose.apis import MMPoseInferencer  # type: ignore

        self._inferencer = MMPoseInferencer(
            pose2d=str(config), pose2d_weights=str(checkpoint), device=device
        )

    def __call__(self, frame: np.ndarray) -> List[PosePrediction]:
        generator = self._inferencer([frame[:, :, ::-1]])  # convert BGR->RGB
        if hasattr(generator, "__iter__") and not isinstance(generator, dict):
            results = list(generator)
            result = results[0] if results else None
        else:
            result = generator

        keypoints = extract_keypoints_from_result(result)
        return [PosePrediction(kp) for kp in keypoints] if keypoints else []


class Pose2DInferencerAdapter(BaseInferencer):
    """Adapter for older MMPose versions exposing ``Pose2DInferencer``."""

    def __init__(self, config: Path, checkpoint: Path, device: str) -> None:
        try:
            from mmpose.apis import Pose2DInferencer  # type: ignore
        except ImportError:
            from mmpose.apis.inferencers import Pose2DInferencer  # type: ignore

        self._inferencer = Pose2DInferencer(str(config), str(checkpoint), device=device)

    def __call__(self, frame: np.ndarray) -> List[PosePrediction]:
        # Pose2DInferencer expects BGR images.
        generator = self._inferencer(frame)
        if hasattr(generator, "__iter__") and not isinstance(generator, dict):
            results = list(generator)
            result = results[0] if results else None
        else:
            result = generator

        keypoints = extract_keypoints_from_result(result)
        return [PosePrediction(kp) for kp in keypoints] if keypoints else []


def build_inferencer(config: Path, checkpoint: Path, device: str) -> BaseInferencer:
    """Instantiate the best available inferencer implementation."""

    attempt_errors: List[str] = []

    try:
        return MMPoseInferencerAdapter(config, checkpoint, device)
    except Exception as exc:
        attempt_errors.append(f"MMPoseInferencer unavailable: {exc}")

    try:
        return Pose2DInferencerAdapter(config, checkpoint, device)
    except Exception as exc:
        attempt_errors.append(f"Pose2DInferencer unavailable: {exc}")

    raise InferenceError("\n".join(attempt_errors))


# ---------------------------------------------------------------------------
# Result post-processing helpers
# ---------------------------------------------------------------------------


def extract_keypoints_from_result(result: Optional[object]) -> List[np.ndarray]:
    """Extract a ``List[np.ndarray]`` from the raw MMPose result structure."""

    if result is None:
        return []

    # For MMPoseInferencer the data is stored inside ``predictions``.
    if isinstance(result, dict):
        if "predictions" in result and result["predictions"]:
            result = result["predictions"][0]

    if isinstance(result, dict):
        for key in ("preds", "preds_2d", "keypoints", "pred_instances"):
            value = result.get(key)
            if value is None:
                continue
            if key == "pred_instances" and isinstance(value, dict):
                value = value.get("keypoints")
            if value is None:
                continue
            return normalise_prediction_sequence(value)

    return normalise_prediction_sequence(result)


def normalise_prediction_sequence(raw: object) -> List[np.ndarray]:
    """Convert different container types into a list of ``(K, 3)`` numpy arrays."""

    if raw is None:
        return []

    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            return [np.asarray(person) for person in raw]
        if raw.ndim == 2:
            return [np.asarray(raw)]
        return []

    if isinstance(raw, Sequence):
        people: List[np.ndarray] = []
        for person in raw:
            arr = np.asarray(person)
            if arr.ndim == 1:
                continue
            if arr.shape[-1] == 2:  # scores missing -> append dummy column
                scores = np.ones((arr.shape[0], 1), dtype=arr.dtype)
                arr = np.concatenate([arr, scores], axis=1)
            people.append(arr)
        return people

    return []


def draw_keypoints(frame: np.ndarray, predictions: Sequence[PosePrediction], threshold: float = 0.2) -> np.ndarray:
    """Overlay green circles for keypoints with a confidence above ``threshold``."""

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    for person in predictions:
        if person.keypoints.size == 0:
            continue
        for x, y, score in person.keypoints:
            if score < threshold:
                continue
            if not (0 <= x < width and 0 <= y < height):
                continue
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)
    return canvas


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


def run_realtime_loop(
    inferencer: BaseInferencer,
    camera: int,
    display_scale: float,
    score_threshold: float,
    skip_frames: int,
) -> None:
    """Capture frames from the camera, run inference and display the result."""

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {camera}")

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; stopping.")
                break
            frame_index += 1

            if skip_frames and frame_index % (skip_frames + 1) != 1:
                if display_scale != 1.0:
                    display = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
                else:
                    display = frame
                cv2.imshow("SynthPose", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            predictions = inferencer(frame)
            annotated = draw_keypoints(frame, predictions, threshold=score_threshold)
            if display_scale != 1.0:
                annotated = cv2.resize(annotated, None, fx=display_scale, fy=display_scale)

            cv2.imshow("SynthPose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime SynthPose HRNet-48 demo")
    parser.add_argument("--repo", default="stanfordmimi/synthpose-hrnet-48-mmpose", help="Hugging Face repository")
    parser.add_argument("--model-dir", default="./synthpose_model", help="Where to store/download the model snapshot")
    parser.add_argument("--device", default="auto", help="'cpu', 'cuda:0', or 'auto' to detect automatically")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0 = default webcam)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Resize factor for the OpenCV window")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="Keypoint score threshold for drawing")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between inferences")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    try:
        model_root = ensure_model_snapshot(args.repo, args.model_dir)
        config_path, checkpoint_path = discover_config_and_checkpoint(model_root)
    except Exception as exc:
        print(f"Failed to prepare model files: {exc}")
        return 1

    if not config_path or not checkpoint_path:
        print("Could not locate a config (.py) and checkpoint (.pth) inside the snapshot.")
        print("Please supply --model-dir pointing to a folder that contains both files.")
        return 1

    print(f"Using config:    {config_path}")
    print(f"Using checkpoint: {checkpoint_path}")

    try:
        inferencer = build_inferencer(config_path, checkpoint_path, device)
    except InferenceError as exc:
        print("Failed to initialise MMPose. Please ensure mmpose is installed and up to date.")
        print(str(exc))
        return 1

    print("Starting realtime capture. Press 'q' to exit.")
    try:
        run_realtime_loop(
            inferencer,
            camera=args.camera,
            display_scale=args.display_scale,
            score_threshold=args.score_threshold,
            skip_frames=args.skip_frames,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print(f"Runtime error: {exc}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
