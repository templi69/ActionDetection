"""
robot_pipeline.py

Member 3 + Member 4 integration.

Pipeline:

YOLO -> Tracker -> MediaPipe -> ActionBuffer -> VideoMAE
     -> ActionSmoother -> DecisionEngine -> Robot

Features:

- Multiple-person tracking
- Independent action buffer per person
- Independent ActionSmoother per person
- Persistent action between VideoMAE predictions
- Safety-aware DecisionEngine
- VideoMAE fall confidence threshold
- Simulated or Serial robot
- Action visualization above bounding box
"""

import argparse
import cv2

from detection import PersonDetector
from pose_extraction import PoseExtractor
from tracker import CentroidTracker
from visualization import draw_tracked_people

from action_buffer import ActionBuffer
from action_recognizer import VideoMAEActionRecognizer
from action_smoother import ActionSmoother

from robot_interface import (
    SimulatedRobot,
    SerialRobot,
    RobotState
)

from decision_engine import (
    DecisionEngine,
    DecisionConfig
)


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Full action recognition "
            "+ robot integration pipeline"
        )
    )

    p.add_argument("--source", default="0")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--robot", choices=["sim", "serial"], default="sim")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--log", default="robot_run.log")

    return p.parse_args()


# ==========================================================
# ROBOT STATE COLORS
# ==========================================================

STATE_COLORS = {
    RobotState.IDLE: (200, 200, 200),
    RobotState.FOLLOWING: (0, 255, 0),
    RobotState.APPROACHING: (0, 200, 255),
    RobotState.ASSISTING: (0, 165, 255),
    RobotState.STOPPED: (0, 0, 255),
    RobotState.EMERGENCY_STOP: (0, 0, 255),
}


# ==========================================================
# ROBOT STATUS OVERLAY
# ==========================================================

def overlay_robot_status(frame, robot, decision):

    text = (
        f"ROBOT: {robot.status.state.value}"
        f" | {decision.get('decision', '')}"
    )

    color = STATE_COLORS.get(robot.status.state, (255, 255, 255))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)

    cv2.putText(
        frame, text, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )

    return frame


# ==========================================================
# PERSON ACTION LABEL
# ==========================================================

def draw_action_label(frame, person):
    """
    Draw persistent action information above the person's
    bounding box, positioned high enough that it doesn't
    cover the person's body.
    """

    x1, y1, x2, y2 = person["bbox"]

    person_id = person["id"]
    action = person.get("action", "unknown")
    confidence = float(person.get("action_confidence", 0.0) or 0.0)

    text = f"ID {person_id}: {action} {confidence:.2f}"

    text_x = int(x1)
    text_y = max(20, int(y1) - 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    bg_x1 = text_x
    bg_y1 = max(0, text_y - text_height - 6)
    bg_x2 = text_x + text_width + 8
    bg_y2 = text_y + baseline + 3

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

    cv2.putText(
        frame, text, (text_x + 4, text_y),
        font, font_scale, (0, 255, 255), thickness,
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run(args):

    # ------------------------------------------------------
    # VIDEO SOURCE
    # ------------------------------------------------------

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    # ------------------------------------------------------
    # MEMBER 2
    # ------------------------------------------------------

    detector = PersonDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )

    pose_extractor = PoseExtractor()
    tracker = CentroidTracker()

    # ------------------------------------------------------
    # MEMBER 3
    # ------------------------------------------------------

    SEQUENCE_LENGTH = 16
    INFERENCE_INTERVAL = 8

    action_buffer = ActionBuffer(sequence_length=SEQUENCE_LENGTH)

    action_recognizer = VideoMAEActionRecognizer(
        sequence_length=SEQUENCE_LENGTH,
        device=args.device,
    )

    # One smoother per person.
    smoothers = {}

    # Persistent action state per person.
    #
    # Prevents:
    #   walking -> unknown -> unknown
    # between VideoMAE predictions.
    last_actions = {}

    # ------------------------------------------------------
    # MEMBER 4
    # ------------------------------------------------------

    if args.robot == "sim":
        robot = SimulatedRobot(log_path=args.log)
    else:
        robot = SerialRobot(port=args.port)

    engine = DecisionEngine(
        robot,
        frame_width=width,
        config=DecisionConfig(
            proximity_area_ratio=0.9,
            action_fall_confidence=0.60,
        ),
    )

    frame_idx = 0

    # ------------------------------------------------------
    # STARTUP MESSAGE
    # ------------------------------------------------------

    print()
    print("======================================")
    print("FULL ROBOT ACTION PIPELINE")
    print("======================================")
    print("YOLO:              ENABLED")
    print("MediaPipe:         ENABLED")
    print("Tracker:           ENABLED")
    print("ActionBuffer:      ENABLED")
    print("VideoMAE:          ENABLED")
    print("Smoother:          ENABLED")
    print("Persistent Action: ENABLED")
    print("Safety Layer:      ENABLED")
    print("Fall Confidence:   0.60")
    print("DecisionEngine:    ENABLED")
    print("Robot:             ENABLED")
    print("======================================")
    print()

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print("Video source ended.")
                break

            frame_idx += 1

            # ==================================================
            # 1. YOLO PERSON DETECTION
            # ==================================================

            detections = detector.detect(frame)

            # ==================================================
            # 2. TRACK PEOPLE
            # ==================================================

            tracked_people = tracker.update(detections)
            current_ids = set()

            # ==================================================
            # 3. PROCESS EACH PERSON
            # ==================================================

            for person in tracked_people:

                person_id = person["id"]
                current_ids.add(person_id)

                # MediaPipe pose.
                crop, offset = detector.crop_person(frame, person["bbox"])
                person["landmarks"] = pose_extractor.extract(crop, offset)

                # Create smoother for new person.
                if person_id not in smoothers:

                    smoothers[person_id] = ActionSmoother(
                        window_size=7,
                        min_frame_confidence=0.03,
                        stable_confidence_threshold=0.15,
                    )

                    print(f"[Action] Created state for Person {person_id}")

                # Create persistent action state.
                if person_id not in last_actions:
                    last_actions[person_id] = {"action": "unknown", "confidence": 0.0}

                # Add current frame to person's buffer.
                sequence = action_buffer.add_frame(person_id, frame)

                # Use the LAST known action instead of resetting to "unknown".
                person["action"] = last_actions[person_id]["action"]
                person["action_confidence"] = last_actions[person_id]["confidence"]

                # ==================================================
                # 4. VIDEOMAE INFERENCE
                # ==================================================

                if sequence is not None and frame_idx % INFERENCE_INTERVAL == 0:

                    try:
                        result = action_recognizer.predict(sequence)

                        raw_action = result["action"]
                        raw_confidence = float(result["confidence"])

                        # Smooth prediction.
                        stable_result = smoothers[person_id].update(
                            raw_action, raw_confidence
                        )

                        stable_action = stable_result["action"]
                        stable_confidence = float(stable_result["confidence"])

                        # Save persistent action.
                        last_actions[person_id] = {
                            "action": stable_action,
                            "confidence": stable_confidence,
                        }

                        person["action"] = stable_action
                        person["action_confidence"] = stable_confidence

                        raw_model_action = result.get("raw_action", raw_action)

                        print(
                            f"[Person {person_id}] "
                            f"Raw: {raw_model_action} -> {raw_action} "
                            f"({raw_confidence:.3f}) | "
                            f"Stable: {stable_action} ({stable_confidence:.3f})"
                        )

                    except Exception as e:
                        print(f"[Person {person_id}] VideoMAE error: {e}")

            # ==================================================
            # 5. CLEANUP LOST PEOPLE
            # ==================================================

            known_ids = set(action_buffer.buffers.keys())

            for person_id in known_ids:
                if person_id not in current_ids:
                    action_buffer.remove_person(person_id)
                    smoothers.pop(person_id, None)
                    last_actions.pop(person_id, None)
                    print(f"[Action] Removed Person {person_id}")

            # ==================================================
            # 6. DECISION ENGINE
            # ==================================================

            decision = engine.step(frame_idx, tracked_people)

            # ==================================================
            # 7. VISUALIZATION
            # ==================================================

            annotated = draw_tracked_people(frame.copy(), tracked_people)
            annotated = overlay_robot_status(annotated, robot, decision)

            for person in tracked_people:
                draw_action_label(annotated, person)

            # ==================================================
            # 8. DISPLAY
            # ==================================================

            if not args.no_display:

                cv2.imshow("Robot + VideoMAE", annotated)

                key = cv2.waitKey(1) & 0xFF

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

    print()
    print(f"Processed {frame_idx} frames.")
    print(f"Final robot state: {robot.status.state.value}")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run(parse_args())