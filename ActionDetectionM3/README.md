# Object Detection & Pose Estimation Module

Implements Member 2's responsibilities:
- Human detection (YOLO)
- Pose estimation (MediaPipe)
- Body landmark extraction
- Pose sequence generation
- Skeleton visualization

## Files

| File | Deliverable | What it does |
|---|---|---|
| `detection.py` | Detection module | `PersonDetector` runs YOLO, returns bounding boxes for people, crops them out of the frame |
| `pose_extraction.py` | Pose extraction module | `PoseExtractor` runs MediaPipe Pose on a person crop, returns 33 named landmarks (x, y, z, visibility) in full-frame coordinates |
| `tracker.py` | (supports sequences) | `CentroidTracker` gives each person a stable ID across frames using nearest-centroid matching |
| `sequence.py` | Pose sequence generation | `PoseSequenceBuilder` collects per-frame landmarks into a sequence per person ID, exports to JSON |
| `visualization.py` | Skeleton visualization | Draws bounding boxes + skeleton lines/joints per person, color-coded by ID |
| `main.py` | Ties it all together | Full pipeline you actually run: video/webcam in → annotated video + JSON out |

## Setup

```bash
pip install -r requirements.txt
```

First run will auto-download `yolov8n.pt` (~6MB, nano model — fast on CPU).
Swap `--model yolov8s.pt` or `yolov8m.pt` for better accuracy if you have GPU.

## Run

```bash
# Webcam
python main.py --source 0

# Video file
python main.py --source input.mp4 --output annotated.mp4 --json poses.json

# Headless (e.g. on a server, no display window)
python main.py --source input.mp4 --no-display
```

## Output

- **Annotated video** (`output_video.mp4`): original footage with bounding boxes,
  person IDs, confidence scores, and skeleton overlays drawn.
- **Pose sequence JSON** (`pose_sequences.json`): structured as
  ```json
  {
    "0": [
      {"frame": 0, "bbox": [x1, y1, x2, y2], "landmarks": [{"name": "NOSE", "x": .., "y": .., "z": .., "visibility": ..}, ...]},
      {"frame": 1, "bbox": [...], "landmarks": [...]}
    ],
    "1": [ ... ]
  }
  ```
  Keys are tracked person IDs; each entry is that person's landmark sequence
  over time. This is what downstream teammates (e.g. action recognition,
  animation retargeting, VR avatar driving) would consume.

## Design notes / talking points for your report or defense

- **Why crop before running MediaPipe?** MediaPipe Pose assumes roughly one
  centered person per image. Running YOLO first and cropping each detected
  person lets the pipeline handle **multiple people** in one frame — plain
  MediaPipe alone can't do that.
- **Why a tracker?** Detection + pose alone gives you a skeleton per frame,
  not a *sequence*. The centroid tracker assigns stable IDs so landmarks
  across frames belong to the same person over time, which is what makes
  the data usable for something like action/gesture recognition.
- **Tuning knobs:** `--conf` (detection sensitivity), tracker's
  `max_distance`/`max_missed_frames` (occlusion tolerance), MediaPipe's
  `model_complexity` (0=fastest, 2=most accurate).
- **Known limitation:** the centroid tracker is intentionally simple (no
  Kalman filter, no re-ID embeddings) — good enough for a handful of people
  at moderate FPS, but it can swap IDs during heavy occlusion or fast
  crossing. Worth noting as a limitation/future work in your report.
