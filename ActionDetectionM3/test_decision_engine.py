"""
test_decision_engine.py
Member 4 - Deliverable: Test end-to-end system (decision/logic layer)
Exercises the decision engine + simulated robot using synthetic pose data,
no camera, YOLO, or MediaPipe required. Run with:

    pytest test_decision_engine.py -v
"""

from robot_interface import SimulatedRobot, RobotState
from decision_engine import DecisionEngine, DecisionConfig


def make_person(pid, bbox, landmarks=None, action="unknown"):
    return {"id": pid, "bbox": bbox, "landmarks": landmarks, "action": action, "action_confidence": 1.0}


def make_landmarks(overrides):
    """Builds a small standing-pose landmark set, then applies overrides by name."""
    base = {
        "LEFT_SHOULDER": (300, 200), "RIGHT_SHOULDER": (340, 200),
        "LEFT_HIP": (305, 350), "RIGHT_HIP": (335, 350),
        "LEFT_WRIST": (280, 320), "RIGHT_WRIST": (360, 320),
    }
    base.update(overrides)
    return [
        {"name": name, "x": xy[0], "y": xy[1], "z": 0.0, "visibility": 0.9}
        for name, xy in base.items()
    ]


def run_frames(engine, actions_over_time, bbox=(250, 150, 400, 400)):
    """Feeds a sequence of per-frame action labels for a single person id=1."""
    last_decision = None
    for i, action in enumerate(actions_over_time):
        people = [make_person(1, bbox, landmarks=make_landmarks({}), action=action)]
        last_decision = engine.step(i, people)
    return last_decision


def test_idle_when_no_one_present():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640)
    decision = engine.step(0, [])
    assert decision["decision"] == "idle"
    assert robot.status.state == RobotState.IDLE


def test_follows_walking_person_after_debounce():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640, config=DecisionConfig(debounce_frames=3))
    decision = run_frames(engine, ["walking"] * 5)
    assert decision["decision"] == "follow"
    assert robot.status.state == RobotState.FOLLOWING


def test_approaches_on_waving():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640, config=DecisionConfig(debounce_frames=3))
    decision = run_frames(engine, ["waving"] * 5)
    assert decision["decision"] == "approach_assist"
    assert robot.status.state == RobotState.APPROACHING


def test_emergency_stop_on_fall_geometry():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640)
    # Torso roughly horizontal -> should trip the fall heuristic regardless of "action" label
    fallen_landmarks = make_landmarks({
        "LEFT_SHOULDER": (200, 300), "RIGHT_SHOULDER": (200, 340),
        "LEFT_HIP": (400, 305), "RIGHT_HIP": (400, 335),
    })
    people = [make_person(1, (150, 280, 420, 350), landmarks=fallen_landmarks, action="walking")]
    decision = engine.step(0, people)
    assert decision["decision"] == "emergency_stop"
    assert robot.status.state == RobotState.EMERGENCY_STOP
    assert robot.in_emergency


def test_emergency_latches_until_cleared():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640)
    robot.emergency_stop(reason="test trigger")
    decision = engine.step(0, [make_person(1, (250, 150, 400, 400), action="walking")])
    assert decision["decision"] == "holding_emergency_stop"
    robot.clear_emergency()
    assert not robot.in_emergency


def test_proximity_triggers_emergency_stop():
    robot = SimulatedRobot()
    engine = DecisionEngine(robot, frame_width=640, config=DecisionConfig(proximity_area_ratio=0.2))
    # Huge bbox relative to frame -> "too close"
    people = [make_person(1, (0, 0, 620, 350), action="walking")]
    decision = engine.step(0, people)
    assert decision["decision"] == "emergency_stop"


def test_target_lost_triggers_stop():
    robot = SimulatedRobot()
    engine = DecisionEngine(
        robot, frame_width=640,
        config=DecisionConfig(debounce_frames=2, lost_tracking_grace=2),
    )
    run_frames(engine, ["walking", "walking", "walking"])  # establishes target id=1, last seen at frame 2
    engine.step(3, [])  # person vanishes (missing 1 frame, still within grace=2)
    decision = engine.step(5, [])  # missing for 3 frames now, beyond grace period
    assert decision["decision"] == "stop_target_lost"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
