"""Real-time SynthPose MMPose webcam demo.

This script adapts the official Hugging Face demo (which expects a video file)
so that it runs continuously on webcam input.  It keeps the following goals in
mind:

* CPU friendly – works with MMCV-lite builds and gracefully falls back when GPU
  is unavailable.
* Minimal dependencies beyond OpenCV, NumPy, torch, huggingface_hub and MMPose.
* Modular – individual helper functions are easy to reuse in other
  applications/servers.

Usage example (after installing requirements from requirements.txt):

    python pose_stream_server/synthpose_mmpose_webcam_server.py \
        --model-dir ./synthpose_model_cache --camera 0
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
import time
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

try:
    from mmpose.apis import MMPoseInferencer
except Exception as exc:  # pragma: no cover - informative message for users
    raise ImportError(
        "MMPose could not be imported. Please install mmpose>=1.2 and its "
        "dependencies before running this script."
    ) from exc


KeypointArray = Sequence[Sequence[float]]


def download_model_if_needed(repo_id: str, target_dir: Path) -> Path:
    """Download model snapshot into *target_dir* if the directory is empty.

    Parameters
    ----------
    repo_id:
        Hugging Face repository identifier (e.g. ``stanfordmimi/synthpose-hrnet-48-mmpose``).
    target_dir:
        Directory on disk where the snapshot should live.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    # A quick heuristic: if the directory is empty, trigger a download.
    if any(target_dir.iterdir()):
        return target_dir

    print(f"Downloading SynthPose weights from '{repo_id}' to '{target_dir}' ...")
    snapshot_download(repo_id=repo_id, local_dir=target_dir)
    print("Download completed.")
    return target_dir


def discover_model_files(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return potential (config_path, weight_path) inside *model_dir*.

    The Stanford SynthPose repo ships with a single config python file and a
    ``.pth`` checkpoint – we simply pick the first match for each.
    """

    config_path = next(model_dir.rglob("*.py"), None)
    weight_path = next(model_dir.rglob("*.pth"), None)
    return config_path, weight_path


def create_inferencer(
    config_path: Path,
    weight_path: Path,
    device: str,
) -> MMPoseInferencer:
    """Instantiate the MMPose inferencer."""

    print(f"Loading MMPose inferencer on device '{device}' ...")
    inferencer = MMPoseInferencer(
        pose2d=str(config_path),
        pose2d_weights=str(weight_path),
        device=device,
    )
    return inferencer


def webcam_frames(
    source: int | str,
    queue_sleep: float = 0.001,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Yield timestamped frames from a webcam/video source."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera/video source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield time.time(), frame
            # avoid busy-waiting in case of very fast camera read
            if queue_sleep:
                time.sleep(queue_sleep)
    finally:
        cap.release()


def _consume_generator(result) -> Optional[dict]:
    """MMPoseInferencer may return a generator – grab its first element."""

    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, Iterable):
        iterator = iter(result)
        return next(iterator, None)
    return None


def _extract_keypoints(result_dict) -> Optional[KeypointArray]:
    """Retrieve the keypoints array from the inferencer output."""

    if not result_dict:
        return None

    for key in ("preds", "preds_2d", "keypoints", "preds_xywh"):
        value = result_dict.get(key)
        if value is not None:
            return value

    instances = result_dict.get("instances")
    if isinstance(instances, dict):
        for key in ("preds", "keypoints", "pred_keypoints"):
            value = instances.get(key)
            if value is not None:
                return value
    return None


def infer_keypoints(
    inferencer: MMPoseInferencer,
    frame_bgr: np.ndarray,
    tmp_dir: Path,
) -> Optional[KeypointArray]:
    """Run SynthPose on *frame_bgr* and return keypoints."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        raw = inferencer([frame_rgb], pred_out_dir=None, vis_out_dir=None)
        result_dict = _consume_generator(raw)
        if result_dict is None:
            return None
        return _extract_keypoints(result_dict)
    except TypeError:
        # Some MMPose versions only accept file paths (no numpy arrays).
        tmp_path = tmp_dir / "frame.jpg"
        cv2.imwrite(str(tmp_path), frame_bgr)
        raw = inferencer(str(tmp_path), pred_out_dir=None, vis_out_dir=None)
        result_dict = _consume_generator(raw)
        if result_dict is None:
            return None
        return _extract_keypoints(result_dict)


def draw_keypoints(
    frame: np.ndarray,
    keypoints: Optional[KeypointArray],
    score_threshold: float = 0.25,
) -> np.ndarray:
    """Overlay keypoints on a frame and return a copy."""

    if keypoints is None:
        return frame

    canvas = frame.copy()
    people: List[np.ndarray]
    if isinstance(keypoints, np.ndarray):
        if keypoints.ndim == 2:
            people = [keypoints]
        elif keypoints.ndim == 3:
            people = [kp for kp in keypoints]
        else:
            return canvas
    elif isinstance(keypoints, Sequence):
        people = [np.asarray(kp) for kp in keypoints]
    else:
        return canvas

    height, width = canvas.shape[:2]
    for person in people:
        if person.ndim != 2 or person.shape[1] < 2:
            continue
        xs = person[:, 0].astype(int)
        ys = person[:, 1].astype(int)
        if person.shape[1] >= 3:
            scores = person[:, 2]
        else:
            scores = np.ones_like(xs, dtype=float)
        for x, y, score in zip(xs, ys, scores):
            if score < score_threshold:
                continue
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)
    return canvas


def flatten_keypoints(keypoints: Optional[KeypointArray]) -> List[float]:
    """Flatten ``(x, y, score)`` keypoints to a simple list."""

    if keypoints is None:
        return []
    array = np.asarray(keypoints)
    if array.ndim == 3:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim != 2:
        return []
    return array.flatten().astype(float).tolist()


def select_device(user_choice: str) -> str:
    """Pick a device string based on user preference and CUDA availability."""

    if user_choice.lower() != "auto":
        return user_choice
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time SynthPose webcam server")
    parser.add_argument(
        "--repo",
        type=str,
        default="stanfordmimi/synthpose-hrnet-48-mmpose",
        help="Hugging Face repo id containing the SynthPose config + checkpoint.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./synthpose_model"),
        help="Local directory where the model snapshot will be stored.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index or video path. Integers are treated as OpenCV camera indices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto', 'cpu', 'cuda:0', ...",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Keypoint confidence threshold for drawing.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Resize factor for the preview window.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("synthpose_webcam_log.csv"),
        help="Optional CSV file to log flattened keypoints.",
    )
    return parser.parse_args(argv)


def resolve_camera_arg(camera_arg: str) -> int | str:
    """Interpret camera argument as int when possible."""

    with contextlib.suppress(ValueError):
        return int(camera_arg)
    return camera_arg


def open_csv_logger(csv_path: Path):
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame_index", "num_values", "flattened_keypoints"])
    return csv_file, writer


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model_dir = download_model_if_needed(args.repo, args.model_dir)
    config_path, weight_path = discover_model_files(model_dir)
    if config_path is None or weight_path is None:
        raise FileNotFoundError(
            "Could not locate config (.py) and checkpoint (.pth) files in the model directory."
        )

    inferencer = create_inferencer(config_path, weight_path, device=device)

    camera_source = resolve_camera_arg(args.camera)

    tmp_dir = Path(tempfile.mkdtemp(prefix="synthpose_frames_"))
    print(f"Temporary directory for frame dumps: {tmp_dir}")

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file, csv_writer = open_csv_logger(args.log_csv)

    try:
        for idx, (timestamp, frame) in enumerate(webcam_frames(camera_source)):
            keypoints = infer_keypoints(inferencer, frame, tmp_dir)
            if csv_writer is not None:
                values = flatten_keypoints(keypoints)
                csv_writer.writerow([timestamp, idx, len(values), values])
                csv_file.flush()

            visual = draw_keypoints(frame, keypoints, score_threshold=args.score_threshold)
            cv2.putText(
                visual,
                f"frame {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if args.display_scale != 1.0:
                visual = cv2.resize(
                    visual,
                    dsize=None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                )
            cv2.imshow("SynthPose webcam", visual)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exit requested by user.")
                break
    finally:
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()
        # Remove temporary directory contents.
        for item in tmp_dir.glob("*"):
            with contextlib.suppress(OSError):
                item.unlink()
        with contextlib.suppress(OSError):
            tmp_dir.rmdir()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
