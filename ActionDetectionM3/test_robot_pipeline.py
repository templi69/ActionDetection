"""
test_robot_pipeline.py

Member 4 - Integration tests

Feeds synthetic action sequences directly into DecisionEngine
+ SimulatedRobot, bypassing YOLO / MediaPipe / VideoMAE entirely.

This proves the decision -> robot command chain works correctly
before we trust it with live camera + VideoMAE input.

Run:
    python -m pytest test_robot_pipeline.py -v
"""

import os
import pytest

from robot_interface import SimulatedRobot, RobotState
from decision_engine import DecisionEngine, DecisionConfig


FRAME_WIDTH = 640


# ==========================================================
# HELPERS
# ==========================================================

def make_person(
    person_id=0,
    action="unknown",
    confidence=0.0,
    x1=270,
    y1=100,
    x2=370,
    y2=400,
):
    """Build a synthetic tracked_people entry."""
    return {
        "id": person_id,
        "bbox": (x1, y1, x2, y2),
        "landmarks": None,
        "action": action,
        "action_confidence": confidence,
    }


def make_engine(tmp_log, **config_overrides):
    robot = SimulatedRobot(log_path=tmp_log)
    cfg = DecisionConfig(
        proximity_area_ratio=0.9,
        action_fall_confidence=0.60,
        **config_overrides,
    )
    engine = DecisionEngine(robot, frame_width=FRAME_WIDTH, config=cfg)
    return engine, robot


def feed_action(engine, action, confidence, frames=6, person_id=0, **bbox):
    """
    Step the engine `frames` times with the same action/confidence.

    debounce_frames defaults to 5, so 6 frames guarantees the
    action becomes "stable" from the DecisionEngine's point of view.
    """
    decision = None
    for i in range(frames):
        person = make_person(
            person_id=person_id,
            action=action,
            confidence=confidence,
            **bbox,
        )
        decision = engine.step(i + 1, [person])
    return decision


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "robot_run.log")


# ==========================================================
# TESTS
# ==========================================================

def test_walking_causes_following(log_path):
    engine, robot = make_engine(log_path)

    decision = feed_action(engine, "walking", confidence=0.9)

    assert decision["decision"] == "follow"
    assert robot.status.state == RobotState.FOLLOWING


def test_waving_causes_approaching(log_path):
    engine, robot = make_engine(log_path)

    decision = feed_action(engine, "waving", confidence=0.9)

    assert decision["decision"] == "approach_assist"
    assert robot.status.state == RobotState.APPROACHING


def test_falling_causes_emergency_stop(log_path):
    engine, robot = make_engine(log_path)

    person = make_person(action="falling", confidence=0.75)
    decision = engine.step(1, [person])

    assert decision["decision"] == "emergency_stop"
    assert decision["reason"] == "videomae_fall"
    assert robot.status.state == RobotState.EMERGENCY_STOP
    assert robot.in_emergency is True


def test_low_conf_fall_is_ignored(log_path):
    engine, robot = make_engine(log_path)

    # Below the 0.60 threshold -> should NOT trigger emergency stop.
    person = make_person(action="falling", confidence=0.40)
    decision = engine.step(1, [person])

    assert decision["decision"] != "emergency_stop"
    assert robot.in_emergency is False


def test_proximity_causes_emergency(log_path):
    engine, robot = make_engine(log_path)

    # Bounding box covers almost the whole frame -> too close.
    person = make_person(
        action="unknown",
        confidence=0.0,
        x1=0, y1=0, x2=FRAME_WIDTH, y2=480,
    )
    decision = engine.step(1, [person])

    assert decision["decision"] == "emergency_stop"
    assert decision["reason"] == "proximity"
    assert robot.in_emergency is True


def test_emergency_latches(log_path):
    engine, robot = make_engine(log_path)

    # Trigger an emergency via falling.
    fallen = make_person(action="falling", confidence=0.9)
    engine.step(1, [fallen])
    assert robot.in_emergency is True

    # Even if the person is now walking, the emergency should
    # remain latched until explicitly cleared.
    walking = make_person(action="walking", confidence=0.9)
    decision = engine.step(2, [walking])

    assert decision["decision"] == "holding_emergency_stop"
    assert robot.in_emergency is True


def test_clear_emergency(log_path):
    engine, robot = make_engine(log_path)

    fallen = make_person(action="falling", confidence=0.9)
    engine.step(1, [fallen])
    assert robot.in_emergency is True

    robot.clear_emergency()
    assert robot.in_emergency is False

    # Robot should now be able to resume normal decisions.
    decision = feed_action(engine, "walking", confidence=0.9, frames=6)
    assert decision["decision"] == "follow"
    assert robot.status.state == RobotState.FOLLOWING


def test_tracking_loss_stops_robot(log_path):
    engine, robot = make_engine(log_path, lost_tracking_grace=2)

    # Establish person 0 as the primary target by walking.
    feed_action(engine, "walking", confidence=0.9, person_id=0, frames=6)
    assert robot.status.state == RobotState.FOLLOWING

    # Person disappears for more frames than lost_tracking_grace.
    # "stop_target_lost" only fires on the single step where the grace
    # period is first exceeded -- after that the primary target is
    # cleared and later steps just report "idle". So we check across
    # all steps rather than only the last one.
    decisions = []
    for i in range(7, 12):
        decisions.append(engine.step(i, []))  # no people tracked

    outcomes = [d["decision"] for d in decisions]
    assert "stop_target_lost" in outcomes
    assert robot.status.state == RobotState.IDLE