"""
real_time_synthpose.py

Ready-to-run script for real-time pose estimation using the SynthPose (HRNet-48) model
from Hugging Face / MMPose. Features:
- OpenCV webcam capture
- MMPoseInferencer integration (uses model you point to)
- GPU / CPU fallback
- Threaded capture + inference pipeline for smooth display
- Visualization overlay on frames
- Logging of keypoints to CSV (timestamp, frame_index, flattened keypoints)
- Graceful handling if inferencer only accepts filenames (falls back to temp file per frame)

Usage:
  python real_time_synthpose.py --repo stanfordmimi/synthpose-hrnet-48-mmpose --device auto

Notes:
- You must have MMPose and its dependencies installed. See MMPose docs if you run into import errors.
- The script tries to call MMPoseInferencer with numpy frames. If that fails, it falls back to writing a temporary image file per frame and passing the filename.
"""

import argparse
import os
import sys
import time
import threading
import queue
import csv
import tempfile
import traceback
from pathlib import Path
import cv2
import torch
from huggingface_hub import snapshot_download
try:
    
    from mmpose.apis import MMPoseInferencer
except Exception:
    MMPoseInferencer = None


def download_model(repo_id: str, local_dir: str):
    """Download model snapshot from Hugging Face to local_dir using snapshot_download.
    If snapshot_download isn't available or fails, we just assume the repo id is a local dir.
    """
    if Path(local_dir).exists():
        return local_dir
    try:
        print(f"Downloading model snapshot for {repo_id} to {local_dir} ...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print("Download complete.")
        return local_dir
    except Exception as e:
        print("Warning: snapshot_download failed or huggingface_hub not available:", e)
        print("If you already have the model files locally, set --model-dir to that path.")
        return local_dir


class FrameProducer(threading.Thread):
    def __init__(self, src=0, queue_size=4):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {src}")
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame read failed; stopping producer")
                self.stop()
                break
            # keep only most recent frame
            try:
                if self.q.full():
                    # drop oldest
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put_nowait((time.time(), frame))
            except queue.Full:
                pass

    def read(self, timeout=0.01):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


def draw_keypoints(frame, preds, threshold=0.25):
    """Draw keypoints returned by MMPose. preds expected as Nx3 array-like: (x, y, score)
    or a list of arrays per person. We'll handle common shapes.
    """
    try:
        import numpy as np
    except Exception:
        return frame

    if preds is None:
        return frame

    # If preds is a list of persons, iterate
    people = preds
    if isinstance(preds, (list, tuple)) and len(preds) > 0 and hasattr(preds[0], '__len__'):
        # might be list of arrays per person
        people = preds
    else:
        people = [preds]

    h, w = frame.shape[:2]
    for person in people:
        arr = None
        try:
            arr = person
            # try to convert to numpy
            import numpy as np
            arr = np.array(arr)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        # If scores included, shape = (N,3). If only x,y then no scores
        if arr.shape[1] == 2:
            xs = arr[:, 0].astype(int)
            ys = arr[:, 1].astype(int)
            scores = [1.0] * len(xs)
        else:
            xs = arr[:, 0].astype(int)
            ys = arr[:, 1].astype(int)
            scores = arr[:, 2]

        for x, y, s in zip(xs, ys, scores):
            if s >= threshold and 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    return frame


def flatten_keypoints(preds):
    """Return a flat list of numbers from preds (x,y,score per keypoint) or empty list if none."""
    if preds is None:
        return []
    import numpy as np
    if isinstance(preds, (list, tuple)) and len(preds) > 0 and hasattr(preds[0], '__len__'):
        # assume first person only for logging
        arr = np.array(preds[0])
    else:
        arr = np.array(preds)
    if arr.ndim == 2:
        return arr.flatten().tolist()
    return []


def main():
    parser = argparse.ArgumentParser(description="Real-time SynthPose HRNet-48 (MMPose) demo")
    parser.add_argument('--repo', type=str, default='stanfordmimi/synthpose-hrnet-48-mmpose', help='Hugging Face repo id or local model dir')
    parser.add_argument('--model-dir', type=str, default='./synthpose_model', help='Local directory to store/download model')
    parser.add_argument('--device', type=str, default='auto', help="Device to run on: 'auto', 'cpu', or 'cuda:0'")
    parser.add_argument('--camera', type=int, default=0, help='Camera index or video file path')
    parser.add_argument('--out-csv', type=str, default='synthpose_log.csv', help='CSV file to log keypoints')
    parser.add_argument('--display-scale', type=float, default=1.0, help='Scale factor for display window')
    parser.add_argument('--skip-frames', type=int, default=0, help='Skip N frames between inferences (0 = process every frame)')
    parser.add_argument('--score-threshold', type=float, default=0.25, help='Keypoint score threshold to visualize')
    args = parser.parse_args()

    # device selection
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # Check mmpose availability
    if MMPoseInferencer is None:
        print('\nERROR: MMPoseInferencer import failed. You must install MMPose and huggingface_hub.\n')
        print('Quick hints:')
        print('  pip install huggingface_hub')
        print('  Follow MMPose installation instructions: https://mmpose.org/en/latest/getting_started.html')
        print('\nAfter installing, re-run this script.')
        sys.exit(1)

    # Download model snapshot (best-effort)
    model_local_dir = args.model_dir
    download_model(args.repo, model_local_dir)

    # Instantiate inferencer: use the typical config/weight names that the HF repo contains.
    # If repo layout differs, user can set model_dir to point to correct files.
    # We'll attempt to autodiscover config and weight files in the model_local_dir.
    config_path = None
    weight_path = None
    for p in Path(model_local_dir).rglob('*.py'):
        name = p.name
        if 'inference' in name or 'inference' in str(p):
            config_path = str(p)
            break
    for p in Path(model_local_dir).rglob('*.pth'):
        weight_path = str(p)
        break

    if config_path is None or weight_path is None:
        print('Warning: Could not automatically find config (.py) and weight (.pth) in', model_local_dir)
        print('You can still proceed by editing the script to pass the correct files.')

    print('Config:', config_path)
    print('Weights:', weight_path)

    try:
        inferencer = MMPoseInferencer(pose2d=config_path, pose2d_weights=weight_path, device=device)
    except Exception as e:
        print('Failed to create MMPoseInferencer:', e)
        traceback.print_exc()
        print('Please ensure the config and weight paths are correct and that MMPose is properly installed.')
        sys.exit(1)

    # Start frame producer
    producer = FrameProducer(src=args.camera)
    producer.start()

    # Prepare CSV logging
    csv_file = open(args.out_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    header = ['timestamp', 'frame_index'] + [f'kp_{i}_{c}' for i in range(2000) for c in ('x', 'y', 's')]
    # We don't know how many keypoints; we'll write header minimally. We'll write rows with variable length.
    csv_writer.writerow(['timestamp', 'frame_index', 'num_keypoints', 'keypoints_flat'])
    csv_file.flush()

    frame_idx = 0
    last_infer_time = 0
    temp_dir = tempfile.mkdtemp(prefix='synthpose_tmp_')
    print('Temporary dir for fallback images:', temp_dir)

    try:
        while True:
            item = producer.read(timeout=0.5)
            if item is None:
                # no frame available yet
                if not producer.stopped:
                    continue
                else:
                    break

            ts, frame = item
            frame_idx += 1

            if args.skip_frames and (frame_idx % (args.skip_frames + 1) != 1):
                # optionally skip processing for performance
                # we still display latest frame
                disp = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
                cv2.imshow('SynthPose (skipping frames)', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Prepare input frame (convert BGR->RGB if needed)
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Try to call inferencer with numpy input. If unsupported, fall back to temp file approach.
            preds = None
            start = time.time()
            try:
                # The MMPoseInferencer may return a generator for video inputs; for single-image inference
                # it may return a list of result dicts. We'll carefully handle the return type.
                res_gen = inferencer([input_frame], pred_out_dir=None, vis_out_dir=None)
                # res_gen may be a generator or a list
                if hasattr(res_gen, '__iter__') and not isinstance(res_gen, dict):
                    # take first result
                    try:
                        res_list = list(res_gen)
                        if len(res_list) > 0:
                            r0 = res_list[0]
                        else:
                            r0 = None
                    except TypeError:
                        # not listable; maybe a generator that yields dicts
                        r0 = next(res_gen, None)
                else:
                    r0 = res_gen

                # r0 expected to be a dict with keys like 'preds' or 'preds_2d'
                if isinstance(r0, dict):
                    if 'preds' in r0:
                        preds = r0['preds']
                    elif 'preds_2d' in r0:
                        preds = r0['preds_2d']
                    elif 'keypoint' in r0:
                        preds = r0['keypoint']
                    else:
                        # try to inspect
                        for k in ('preds', 'preds_2d', 'keypoint', 'preds_xywh'):
                            if k in r0:
                                preds = r0[k]
                                break
                else:
                    # Unknown format: try to use as-is
                    preds = r0

            except TypeError as e:
                # Probably inferencer doesn't accept numpy arrays. Fall back to temp file approach.
                # Write frame to temp file and call inferencer with filename.
                tmp_path = os.path.join(temp_dir, f'frame_{frame_idx:08d}.jpg')
                cv2.imwrite(tmp_path, frame)
                try:
                    res_gen = inferencer(tmp_path, pred_out_dir=None, vis_out_dir=None)
                    if hasattr(res_gen, '__iter__'):
                        res_list = list(res_gen)
                        r0 = res_list[0] if len(res_list) > 0 else None
                    else:
                        r0 = res_gen
                    if isinstance(r0, dict):
                        preds = r0.get('preds') or r0.get('preds_2d') or r0.get('keypoint')
                    else:
                        preds = r0
                except Exception as e2:
                    print('Fallback inferencer call failed:', e2)
                    preds = None
            except Exception as e:
                print('Inference failed on frame', frame_idx, ':', e)
                traceback.print_exc()
                preds = None

            infer_time = time.time() - start
            last_infer_time = infer_time

            # Visualization
            vis_frame = frame.copy()
            vis_frame = draw_keypoints(vis_frame, preds, threshold=args.score_threshold)
            cv2.putText(vis_frame, f'Frame: {frame_idx} TPS: {1.0/(infer_time+1e-8):.1f} Inference(s): {infer_time:.3f}s',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            disp = cv2.resize(vis_frame, None, fx=args.display_scale, fy=args.display_scale)
            cv2.imshow('SynthPose Real-time', disp)

            # Logging: write timestamp, frame_index, number of kps and flattened list as one CSV column
            try:
                flat = flatten_keypoints(preds)
                csv_writer.writerow([ts, frame_idx, len(flat) // 3 if flat else 0, flat])
                csv_file.flush()
            except Exception as e:
                print('Logging failed:', e)

            # Check key presses
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        producer.stop()
        csv_file.close()
        cv2.destroyAllWindows()
        print('Exiting. CSV log saved to', args.out_csv)


if __name__ == '__main__':
    main()
