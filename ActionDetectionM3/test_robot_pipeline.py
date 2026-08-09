
"""
test_robot_pipeline.py

Integration tests for:

DecisionEngine
    +
SimulatedRobot

These tests do NOT use YOLO, MediaPipe, or VideoMAE.

They verify that recognized actions and safety conditions
produce the correct robot behavior.
"""

import pytest

from decision_engine import (
    DecisionEngine,
    DecisionConfig
)

from robot_interface import (
    SimulatedRobot,
    RobotState
)


# ==========================================================
# HELPERS
# ==========================================================

def create_robot_and_engine():
    """
    Create a fresh simulated robot and decision engine.
    """

    robot = SimulatedRobot()

    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(

            # Keep this high enough to avoid accidental
            # proximity emergency during normal tests.
            proximity_area_ratio=0.9,

            # VideoMAE fall threshold.
            action_fall_confidence=0.60

        )

    )

    return robot, engine


def make_person(
    person_id=0,
    action="unknown",
    confidence=0.0,
    bbox=(250, 100, 390, 400),
    landmarks=None
):
    """
    Create a tracked-person dictionary in the same
    format used by robot_pipeline.py.
    """

    return {

        "id": person_id,

        "bbox": bbox,

        "landmarks": landmarks,

        "action": action,

        "action_confidence": confidence

    }


def run_action_for_frames(
    engine,
    person,
    start_frame=1,
    frames=5
):
    """
    Feed the same person/action to the DecisionEngine
    for several frames.

    The DecisionEngine uses debounce_frames, so this
    allows an action to become stable.
    """

    decision = None

    for frame_idx in range(

        start_frame,

        start_frame + frames

    ):

        decision = engine.step(

            frame_idx,

            [person]

        )

    return decision


# ==========================================================
# TEST 1
# WALKING -> FOLLOWING
# ==========================================================

def test_walking_causes_following():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="walking",

        confidence=0.90

    )

    decision = run_action_for_frames(

        engine,

        person

    )

    assert decision["decision"] == "follow"

    assert decision["target"] == 0

    assert (
        robot.status.state
        == RobotState.FOLLOWING
    )


# ==========================================================
# TEST 2
# WAVING -> APPROACHING / ASSISTING
# ==========================================================

def test_waving_causes_approaching():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="waving",

        confidence=0.90

    )

    decision = run_action_for_frames(

        engine,

        person

    )

    assert decision["decision"] == (
        "approach_assist"
    )

    assert decision["target"] == 0

    assert (
        robot.status.state
        == RobotState.APPROACHING
    )


# ==========================================================
# TEST 3
# FALLING + HIGH CONFIDENCE
# -> EMERGENCY STOP
# ==========================================================

def test_falling_causes_emergency_stop():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="falling",

        confidence=0.85

    )

    decision = engine.step(

        1,

        [person]

    )

    assert (
        decision["decision"]
        == "emergency_stop"
    )

    assert (
        decision["reason"]
        == "videomae_fall"
        or
        decision["reason"]
        == "videomae_fall"
    )

    assert robot.in_emergency is True

    assert (
        robot.status.state
        == RobotState.EMERGENCY_STOP
    )


# ==========================================================
# TEST 4
# FALLING + LOW CONFIDENCE
# -> SHOULD NOT TRIGGER VIDEOMAE FALL
# ==========================================================

def test_low_confidence_fall_is_ignored():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="falling",

        confidence=0.40

    )

    # Use several frames so debounce can complete.
    decision = run_action_for_frames(

        engine,

        person

    )

    # Low-confidence VideoMAE falling must not
    # directly cause an emergency.
    assert robot.in_emergency is False

    assert (
        robot.status.state
        != RobotState.EMERGENCY_STOP
    )


# ==========================================================
# TEST 5
# PROXIMITY -> EMERGENCY STOP
# ==========================================================

def test_proximity_causes_emergency():

    robot, engine = (
        create_robot_and_engine()
    )

    # Very large bounding box.
    # This should trigger the proximity safety check.
    person = make_person(

        action="unknown",

        confidence=0.0,

        bbox=(20, 20, 620, 470)

    )

    decision = engine.step(

        1,

        [person]

    )

    assert (
        decision["decision"]
        == "emergency_stop"
    )

    assert (
        decision["reason"]
        == "proximity"
    )

    assert robot.in_emergency is True

    assert (
        robot.status.state
        == RobotState.EMERGENCY_STOP
    )


# ==========================================================
# TEST 6
# EMERGENCY MUST REMAIN LATCHED
# ==========================================================

def test_emergency_latches():

    robot, engine = (
        create_robot_and_engine()
    )

    falling_person = make_person(

        action="falling",

        confidence=0.95

    )

    # Trigger emergency.
    engine.step(

        1,

        [falling_person]

    )

    assert robot.in_emergency is True

    # Now give the engine a normal walking person.
    walking_person = make_person(

        action="walking",

        confidence=0.95

    )

    decision = engine.step(

        2,

        [walking_person]

    )

    # Emergency must remain active.
    assert robot.in_emergency is True

    assert (
        decision["decision"]
        == "holding_emergency_stop"
    )


# ==========================================================
# TEST 7
# CLEAR EMERGENCY
# ==========================================================

def test_clear_emergency():

    robot, engine = (
        create_robot_and_engine()
    )

    falling_person = make_person(

        action="falling",

        confidence=0.95

    )

    engine.step(

        1,

        [falling_person]

    )

    assert robot.in_emergency is True

    # Simulate pressing C.
    robot.clear_emergency()

    assert robot.in_emergency is False


# ==========================================================
# TEST 8
# TRACKING LOSS
# ==========================================================


def test_tracking_loss_stops_robot():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="walking",

        confidence=0.90

    )

    # ------------------------------------------------------
    # First establish this person as the target.
    # ------------------------------------------------------

    run_action_for_frames(

        engine,

        person,

        start_frame=1,

        frames=5

    )

    assert (
        robot.status.state
        == RobotState.FOLLOWING
    )


    # ------------------------------------------------------
    # Simulate the person disappearing.
    #
    # We need to check whether the stop decision
    # happens at ANY frame during the grace period.
    # ------------------------------------------------------

    lost_decision_found = False

    for frame_idx in range(6, 30):

        decision = engine.step(

            frame_idx,

            []

        )


        if (
            decision["decision"]
            == "stop_target_lost"
        ):

            lost_decision_found = True

            break


    # ------------------------------------------------------
    # The engine must eventually detect the lost target.
    # ------------------------------------------------------

    assert lost_decision_found is True

    assert (
        robot.status.state
        == RobotState.IDLE
        or
        robot.status.state
        == RobotState.STOPPED
    )



# ==========================================================
# TEST 9
# UNKNOWN ACTION -> IDLE
# ==========================================================

def test_unknown_action_causes_idle():

    robot, engine = (
        create_robot_and_engine()
    )

    person = make_person(

        action="unknown",

        confidence=0.20

    )

    decision = run_action_for_frames(

        engine,

        person

    )

    assert (
        decision["decision"]
        == "idle"
    )

    assert (
        robot.status.state
        == RobotState.IDLE
    )

