"""
main.py
Member 2 - Object Detection & Pose Estimation
Full pipeline: YOLO detection -> crop -> MediaPipe pose -> track -> visualize -> export.

Usage:
    python main.py --source 0                     # webcam
    python main.py --source path/to/video.mp4      # video file
    python main.py --source video.mp4 --output out.mp4 --json poses.json
"""

import argparse
import time

import cv2

from detection import PersonDetector
from pose_extraction import PoseExtractor
from tracker import CentroidTracker
from sequence import PoseSequenceBuilder
from visualization import draw_tracked_people


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + MediaPipe pose pipeline")
    parser.add_argument("--source", default="0",
                         help="Video file path, or camera index (default: 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0")
    parser.add_argument("--output", default="output_video.mp4", help="Path to save annotated video")
    parser.add_argument("--json", default="pose_sequences.json", help="Path to save pose sequence JSON")
    parser.add_argument("--no-display", action="store_true", help="Don't open a live preview window")
    return parser.parse_args()


def run_pipeline(args):
    # Camera index vs video file
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    detector = PersonDetector(model_path=args.model, conf_threshold=args.conf, device=args.device)
    pose_extractor = PoseExtractor()
    tracker = CentroidTracker()
    sequence_builder = PoseSequenceBuilder()

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1. Detect people
            detections = detector.detect(frame)

            # 2. Track them across frames (stable IDs -> sequences, not just per-frame poses)
            tracked_people = tracker.update(detections)

            # 3. Extract pose landmarks per detected person
            for person in tracked_people:
                crop, offset = detector.crop_person(frame, person["bbox"])
                person["landmarks"] = pose_extractor.extract(crop, offset)

            # 4. Build pose sequence for this frame
            sequence_builder.add_frame(frame_idx, tracked_people)

            # 5. Visualize
            annotated = draw_tracked_people(frame.copy(), tracked_people)
            writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Detection + Pose Estimation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        pose_extractor.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    json_path = sequence_builder.save_json(args.json)

    elapsed = time.time() - t0
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} FPS)")
    print(f"People tracked: {sequence_builder.num_people()}")
    print(f"Annotated video saved to: {args.output}")
    print(f"Pose sequences saved to: {json_path}")


if __name__ == "__main__":
    run_pipeline(parse_args())
