# Traffic Lane Vehicle Counter

This repository contains a single-file Python solution and README for counting vehicles in three lanes from a YouTube traffic video. It uses a pretrained YOLO model for detection, a lightweight tracker to maintain consistent IDs across frames, and outputs CSV + annotated video.

---

## Files included

* `traffic_lane_counter.py` — Main Python script (download video, run detection, tracking, lane counting, export CSV, create annotated output video).

---

## Features implemented

* **Video download** from YouTube (pytube) inside the script.
* **Vehicle detection** using a pretrained YOLO (Ultralytics YOLOv8) model (COCO classes filtered to vehicles).
* **Lightweight tracking** (Hungarian matching + IoU / centroid-based) to assign stable IDs to detections across frames and avoid duplicate counts.
* **Lane definition**: divides the frame into **three vertical lanes (left, middle, right)** by default (configurable by percentage). Each tracked vehicle is assigned to a lane by its bounding-box center.
* **Counting**: when a tracked object first appears inside a lane it is counted for that lane. Duplicate counts are prevented by tracking vehicle IDs.
* **Outputs**:

  * CSV file with columns: `vehicle_id`, `lane`, `first_seen_frame`, `first_seen_time_seconds`, `frames_present`, `last_seen_frame`, `last_seen_time_seconds`.
  * Annotated output video with lane boundaries, bounding boxes, IDs and live per-lane counts.
  * Printed summary at end with total counts per lane.

---

## Dependencies

Install using pip (recommended to use a virtual environment):

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` with these lines (or install individually):

```
opencv-python
pytube
ultralytics
pandas
numpy
scipy
lap
```

> Notes:
>
> * `ultralytics` provides YOLO v8 models and is simple to use. If you prefer YOLOv5 or another implementation, adapt the detection section accordingly.
> * `lap` is used for fast linear assignment (Hungarian) via SciPy's `linear_sum_assignment`; SciPy is included by many systems but `lap` was added to improve matching speed. If installation issues arise, you can remove `lap` and rely on `scipy.optimize.linear_sum_assignment`.

---

## How to run

1. Make sure Python 3.8+ is installed and you have a GPU if you want faster inference. CPU works but will be slower.
2. `git clone` this repo (or save the script) and `cd` into the folder.
3. Install dependencies.
4. Run the script:

```bash
python traffic_lane_counter.py --youtube https://www.youtube.com/watch?v=MNn9qKG2UFI --output annotated_out.mp4 --csv results.csv --resize 640
```

Command-line options (short):

```
--youtube  : YouTube video URL (default: the provided video)
--output   : Path for annotated output video
--csv      : Path for CSV output
--model    : YOLO model name / path (default: yolov8n.pt)
--resize   : width to resize frames for faster processing (maintains aspect ratio), default 960 (set lower for real-time on weak CPU)
--max-frames : optional, stop after N frames (for quick demo)
```

---

## How it counts / lane logic

* The script divides the frame's width into three equal columns by default (Left = lane 1, Middle = lane 2, Right = lane 3). You can change this to custom lane polygons by editing the `LANE_POLYGONS` variable in the script.
* For each tracked object we compute its bounding box center. The lane containing that center determines the vehicle's lane.
* When a new tracker ID is created, it's assigned to a lane on the first frame it appears in; that event is recorded to the CSV.
* A per-lane `set` of counted IDs prevents double counting.

---

## Limitations and extensions

* This solution uses a lightweight tracker (Hungarian matching on IoU + centroid) rather than a full SORT/DeepSORT Kalman/appearance based approach. The tracker is compact and easy to adapt; if you need the stronger re-identification performance in heavy occlusion or long-term persistence, integrate **DeepSORT** (many open-source implementations) — switching is straightforward by replacing the `SimpleTracker` class with DeepSORT and adapting detection format.
* Lane geometry is currently vertical thirds; for oblique lanes or camera perspective you should provide polygonal ROIs and a homography to estimate lane crossing lines.

---

## Script: `traffic_lane_counter.py`

Below is the full Python script. Save it as `traffic_lane_counter.py`.

```python
#!/usr/bin/env python3
"""
Traffic Lane Vehicle Counter

Usage example:
    python traffic_lane_counter.py \
      --youtube https://www.youtube.com/watch?v=MNn9qKG2UFI \
      --output annotated_out.mp4 --csv results.csv --resize 960

This script downloads the YouTube video, runs YOLO detection, tracks objects with a
lightweight tracker, assigns lanes (3 vertical lanes by default), counts vehicles per lane,
exports CSV and an annotated video.

Notes:
- Requires `ultralytics` (YOLOv8), `pytube`, `opencv-python`, `pandas`, `numpy`, `scipy`.
- The tracker is a compact IoU + Hungarian assignment tracker implemented here for portability.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from pytube import YouTube

# Try to import ultralytics YOLO; fail with a helpful message
try:
    from ultralytics import YOLO
except Exception as e:
    print("Error importing ultralytics.YOLO. Please install with: pip install ultralytics")
    raise

# For Hungarian matching
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    print("scipy is required (for linear assignment). Install with: pip install scipy")
    raise


# ---------------------- Utility / Tracker ----------------------

def iou(boxA, boxB):
    # boxes [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Track:
    tid: int
    bbox: np.ndarray  # xyxy
    last_seen: int
    hits: int = 0
    skipped_frames: int = 0
    frames_seen: int = 1


class SimpleTracker:
    """
    Lightweight tracker using IoU + Hungarian matching.
    Not a full SORT implementation but works well for medium-density traffic.
    """

    def __init__(self, iou_threshold=0.3, max_skipped_frames=30):
        self.iou_threshold = iou_threshold
        self.max_skipped_frames = max_skipped_frames
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float]], frame_idx: int):
        """
        detections: list of bboxes in xyxy format
        returns list of (track_id, bbox) for current frame
        """
        if len(self.tracks) == 0:
            for d in detections:
                self.tracks.append(Track(self._next_id, np.array(d, dtype=float), frame_idx))
                self._next_id += 1
            return [(t.tid, t.bbox.copy()) for t in self.tracks]

        # build cost matrix (1 - iou)
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - iou(tr.bbox, det)

        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        # update assigned
        for r, c in zip(row_idx, col_idx):
            if r < N and c < M and cost[r, c] < (1.0 - self.iou_threshold):
                # good match
                det = detections[c]
                tr = self.tracks[r]
                tr.bbox = np.array(det, dtype=float)
                tr.last_seen = frame_idx
                tr.hits += 1
                tr.skipped_frames = 0
                tr.frames_seen += 1
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # increment skipped frames for unassigned tracks
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.skipped_frames += 1

        # remove dead tracks
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]

        # create new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(self._next_id, np.array(det, dtype=float), frame_idx))
                self._next_id += 1

        return [(t.tid, t.bbox.copy()) for t in self.tracks]


# ---------------------- Main processing ----------------------

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO ids: car, motorcycle, bus, truck


def download_youtube(url: str, out_path: str = "input_video.mp4") -> str:
    print(f"Downloading video: {url}")
    yt = YouTube(url)
    # choose progressive mp4 stream
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if stream is None:
        raise RuntimeError("Could not find a downloadable mp4 stream for the provided YouTube URL.")
    stream.download(filename=out_path)
    print(f"Saved to {out_path}")
    return out_path


def run(youtube_url: str, output_video: str, csv_path: str, model_name: str = "yolov8n.pt", resize_width: int = 960, max_frames: int = None):
    # 1. download
    input_path = "input_video.mp4"
    if not os.path.exists(input_path):
        download_youtube(youtube_url, out_path=input_path)
    else:
        print(f"Using existing file {input_path}")

    # 2. open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video FPS={fps}, total_frames={total_frames}")

    # 3. load model
    print("Loading YOLO model:", model_name)
    model = YOLO(model_name)

    # 4. prepare output writer
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")
    orig_h, orig_w = frame.shape[:2]

    # compute resized dims
    scale = resize_width / orig_w
    resized_w = int(resize_width)
    resized_h = int(orig_h * scale)

    fourcc = cv2.VideoWriter_fou
```
