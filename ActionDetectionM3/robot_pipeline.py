"""
robot_pipeline.py
Member 4 - Deliverable: End-to-end integration + simulation runner
Hooks the decision engine + simulated (or real) robot onto Member 2's
detection/pose pipeline. Run this instead of main.py once Member 3's
action recognition is available (or use safety.infer_fallback_action
as a stand-in action source, as done here, to test end-to-end today).

Place this file alongside Member 2's detection.py, pose_extraction.py,
tracker.py, visualization.py.

Usage:
    python robot_pipeline.py --source 0
    python robot_pipeline.py --source video.mp4 --no-display
    python robot_pipeline.py --source video.mp4 --robot serial --port /dev/ttyUSB0

Controls while a preview window is open:
    q  -> quit
    c  -> manually clear an emergency stop (simulates an operator override)
"""

import argparse

import cv2

from detection import PersonDetector
from pose_extraction import PoseExtractor
from tracker import CentroidTracker
from visualization import draw_tracked_people

from robot_interface import SimulatedRobot, SerialRobot, RobotState
from decision_engine import DecisionEngine
from safety import infer_fallback_action


def parse_args():
    p = argparse.ArgumentParser(description="Decision logic + robot integration pipeline")
    p.add_argument("--source", default="0")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--robot", choices=["sim", "serial"], default="sim")
    p.add_argument("--port", default="/dev/ttyUSB0", help="Serial port if --robot serial")
    p.add_argument("--log", default="robot_run.log")
    return p.parse_args()


STATE_COLORS = {
    RobotState.IDLE: (200, 200, 200),
    RobotState.FOLLOWING: (0, 255, 0),
    RobotState.APPROACHING: (0, 200, 255),
    RobotState.ASSISTING: (0, 165, 255),
    RobotState.STOPPED: (0, 0, 255),
    RobotState.EMERGENCY_STOP: (0, 0, 255),
}


def overlay_robot_status(frame, robot, decision):
    text = f"ROBOT: {robot.status.state.value} | {decision.get('decision', '')}"
    color = STATE_COLORS.get(robot.status.state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def run(args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

    detector = PersonDetector(model_path=args.model, conf_threshold=args.conf, device=args.device)
    pose_extractor = PoseExtractor()
    tracker = CentroidTracker()

    robot = SimulatedRobot(log_path=args.log) if args.robot == "sim" else SerialRobot(port=args.port)
    from decision_engine import DecisionConfig
    engine = DecisionEngine(robot, frame_width=width, config=DecisionConfig(proximity_area_ratio=0.9))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracked_people = tracker.update(detections)

            for person in tracked_people:
                crop, offset = detector.crop_person(frame, person["bbox"])
                person["landmarks"] = pose_extractor.extract(crop, offset)
                # Stand-in until Member 3's real classifier is wired in:
                person["action"] = infer_fallback_action(person["landmarks"])
                person["action_confidence"] = 1.0

            decision = engine.step(frame_idx, tracked_people)

            annotated = draw_tracked_people(frame.copy(), tracked_people)
            annotated = overlay_robot_status(annotated, robot, decision)

            if not args.no_display:
                cv2.imshow("Decision Logic + Robot Integration", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):          # manual "clear emergency" for testing
                    robot.clear_emergency()

            frame_idx += 1

    finally:
        cap.release()
        pose_extractor.close()
        if args.robot == "sim":
            robot.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Final robot state: {robot.status.state.value}")


if __name__ == "__main__":
    run(parse_args())
