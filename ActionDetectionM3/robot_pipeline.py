
"""
robot_pipeline.py

Member 4 + Member 3 integration:
YOLO -> Person Tracking -> MediaPipe Pose
-> Person-specific temporal buffer -> VideoMAE
-> Action Smoothing -> Decision Engine -> Robot

Usage:
python robot_pipeline.py --source 0
python robot_pipeline.py --source 0 --device cuda
python robot_pipeline.py --source video.mp4 --no-display

Controls:
q -> quit
c -> manually clear emergency stop
"""

import argparse
import time

import cv2

from detection import PersonDetector
from pose_extraction import PoseExtractor
from tracker import CentroidTracker
from visualization import draw_tracked_people

from robot_interface import (
    SimulatedRobot,
    SerialRobot,
    RobotState
)

from decision_engine import (
    DecisionEngine,
    DecisionConfig
)

from safety import infer_fallback_action

# ============================
# MEMBER 3
# ============================

from action_buffer import ActionBuffer
from action_smoother import ActionSmoother
from action_recognizer import VideoMAEActionRecognizer


# ============================
# ARGUMENTS
# ============================

def parse_args():

    p = argparse.ArgumentParser(
        description="YOLO + MediaPipe + VideoMAE + Decision Engine"
    )

    p.add_argument(
        "--source",
        default="0",
        help="Camera index or video path"
    )

    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights path"
    )

    p.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO detection confidence"
    )

    p.add_argument(
        "--device",
        default="cpu",
        help="YOLO device: cpu / cuda / cuda:0"
    )

    p.add_argument(
        "--no-display",
        action="store_true"
    )

    p.add_argument(
        "--robot",
        choices=["sim", "serial"],
        default="sim"
    )

    p.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port if --robot serial"
    )

    p.add_argument(
        "--log",
        default="robot_run.log"
    )

    return p.parse_args()


# ============================
# ROBOT STATUS COLORS
# ============================

STATE_COLORS = {
    RobotState.IDLE: (200, 200, 200),
    RobotState.FOLLOWING: (0, 255, 0),
    RobotState.APPROACHING: (0, 200, 255),
    RobotState.ASSISTING: (0, 165, 255),
    RobotState.STOPPED: (0, 0, 255),
    RobotState.EMERGENCY_STOP: (0, 0, 255),
}


# ============================
# ROBOT STATUS OVERLAY
# ============================

def overlay_robot_status(frame, robot, decision):

    text = (
        f"ROBOT: {robot.status.state.value} | "
        f"{decision.get('decision', '')}"
    )

    color = STATE_COLORS.get(
        robot.status.state,
        (255, 255, 255)
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return frame


# ============================
# ACTION OVERLAY
# ============================

def overlay_action(frame, person):

    if "action" not in person:
        return frame

    action = person.get(
        "action",
        "unknown"
    )

    confidence = person.get(
        "action_confidence",
        0.0
    )

    x1, y1, x2, y2 = map(
        int,
        person["bbox"]
    )

    text = (
        f"ID {person['id']}: "
        f"{action} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )

    return frame


# ============================
# MAIN PIPELINE
# ============================

def run(args):

    # ----------------------------
    # Video source
    # ----------------------------

    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video source: {args.source}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ) or 640

    # ----------------------------
    # Member 2 modules
    # ----------------------------

    detector = PersonDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )

    pose_extractor = PoseExtractor()

    tracker = CentroidTracker()

    # ----------------------------
    # Robot
    # ----------------------------

    robot = (
        SimulatedRobot(log_path=args.log)
        if args.robot == "sim"
        else SerialRobot(port=args.port)
    )

    engine = DecisionEngine(
        robot,
        frame_width=width,
        config=DecisionConfig(
            proximity_area_ratio=0.9
        )
    )

    # ==================================================
    # MEMBER 3 - VideoMAE
    # ==================================================

    SEQUENCE_LENGTH = 16

    INFERENCE_INTERVAL = 8

    # One buffer per tracked person
    action_buffers = {}

    # One smoother per tracked person
    action_smoothers = {}

    print("\n========================================")
    print("Starting integrated robot pipeline")
    print("========================================")
    print("YOLO:       ENABLED")
    print("MediaPipe:  ENABLED")
    print("Tracker:    ENABLED")
    print("VideoMAE:   ENABLED")
    print("Smoothing:  ENABLED")
    print("Robot:      ENABLED")
    print("========================================\n")

    # VideoMAE automatically chooses CUDA
    # when available.
    action_recognizer = VideoMAEActionRecognizer(
        sequence_length=SEQUENCE_LENGTH
    )

    frame_idx = 0

    fps_start = time.time()
    fps_counter = 0
    fps = 0.0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_idx += 1
            fps_counter += 1

            # ==================================================
            # 1. YOLO PERSON DETECTION
            # ==================================================

            detections = detector.detect(frame)

            # ==================================================
            # 2. TRACK PEOPLE
            # ==================================================

            tracked_people = tracker.update(
                detections
            )

            current_ids = set()

            # ==================================================
            # 3. PROCESS EACH PERSON
            # ==================================================

            for person in tracked_people:

                person_id = person["id"]

                current_ids.add(person_id)

                # ------------------------------------------
                # MediaPipe Pose
                # ------------------------------------------

                crop, offset = detector.crop_person(
                    frame,
                    person["bbox"]
                )

                person["landmarks"] = (
                    pose_extractor.extract(
                        crop,
                        offset
                    )
                )

                # ------------------------------------------
                # Create buffer for this person
                # ------------------------------------------

                if person_id not in action_buffers:

                    action_buffers[person_id] = (
                        ActionBuffer(
                            sequence_length=SEQUENCE_LENGTH
                        )
                    )

                    action_smoothers[person_id] = (
                        ActionSmoother(
                            window_size=7,
                            confidence_threshold=0.15
                        )
                    )

                    print(
                        f"[Action] Created buffer "
                        f"for Person {person_id}"
                    )

                # ------------------------------------------
                # Add PERSON CROP to temporal buffer
                # ------------------------------------------

                sequence = action_buffers[
                    person_id
                ].add_frame(
                    person_id,
                    crop
                )

                # ------------------------------------------
                # VideoMAE inference
                # ------------------------------------------

                if (
                    sequence is not None
                    and frame_idx % INFERENCE_INTERVAL == 0
                ):

                    try:

                        result = (
                            action_recognizer.predict(
                                sequence
                            )
                        )

                        raw_action = (
                            result["action"]
                        )

                        raw_confidence = (
                            result["confidence"]
                        )

                        # ----------------------------------
                        # Temporal smoothing
                        # ----------------------------------

                        stable_result = (
                            action_smoothers[
                                person_id
                            ].update(
                                raw_action,
                                raw_confidence
                            )
                        )

                        person["action"] = (
                            stable_result["action"]
                        )

                        person[
                            "action_confidence"
                        ] = stable_result[
                            "confidence"
                        ]

                        print(
                            f"[Person {person_id}] "
                            f"Raw: {raw_action} "
                            f"({raw_confidence:.3f}) | "
                            f"Stable: "
                            f"{person['action']} "
                            f"({person['action_confidence']:.3f})"
                        )

                    except Exception as e:

                        print(
                            f"[Person {person_id}] "
                            f"VideoMAE error: {e}"
                        )

                # ------------------------------------------
                # If VideoMAE hasn't produced an action yet
                # ------------------------------------------

                if "action" not in person:

                    # Safety fallback ONLY until
                    # VideoMAE has enough frames.
                    person["action"] = (
                        infer_fallback_action(
                            person["landmarks"]
                        )
                    )

                    person[
                        "action_confidence"
                    ] = 0.0

            # ==================================================
            # 4. CLEAN UP LOST PERSONS
            # ==================================================

            existing_ids = list(
                action_buffers.keys()
            )

            for person_id in existing_ids:

                if person_id not in current_ids:

                    action_buffers[
                        person_id
                    ].remove_person(
                        person_id
                    )

                    del action_buffers[
                        person_id
                    ]

                    del action_smoothers[
                        person_id
                    ]

                    print(
                        f"[Action] Removed "
                        f"Person {person_id}"
                    )

            # ==================================================
            # 5. DECISION ENGINE
            # ==================================================

            decision = engine.step(
                frame_idx,
                tracked_people
            )

            # ==================================================
            # 6. VISUALIZATION
            # ==================================================

            annotated = draw_tracked_people(
                frame.copy(),
                tracked_people
            )

            # Draw action labels
            for person in tracked_people:

                annotated = overlay_action(
                    annotated,
                    person
                )

            # Robot status
            annotated = overlay_robot_status(
                annotated,
                robot,
                decision
            )

            # FPS
            elapsed = time.time() - fps_start

            if elapsed >= 1.0:

                fps = (
                    fps_counter / elapsed
                )

                fps_counter = 0
                fps_start = time.time()

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # ==================================================
            # 7. DISPLAY
            # ==================================================

            if not args.no_display:

                cv2.imshow(
                    "Robot Action Detection",
                    annotated
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

                if key == ord("c"):

                    robot.clear_emergency()

    finally:

        cap.release()

        pose_extractor.close()

        if args.robot == "sim":

            robot.close()

        if not args.no_display:

            cv2.destroyAllWindows()

    print(
        f"Processed {frame_idx} frames. "
        f"Final robot state: "
        f"{robot.status.state.value}"
    )


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    run(parse_args())

