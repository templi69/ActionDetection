
"""
test_decision_engine.py

Member 4 - Decision Engine Test Suite

Tests the robot decision/logic layer without requiring:

    - Camera
    - YOLO
    - MediaPipe
    - VideoMAE
    - Physical robot

Uses synthetic people and the SimulatedRobot.

Run:

    pytest test_decision_engine.py -v

The tests cover:

    1. Idle behavior
    2. Walking -> Following
    3. Waving -> Assistance
    4. Pose-based fall -> Emergency Stop
    5. VideoMAE falling -> Emergency Stop
    6. Emergency-stop latching
    7. Proximity safety
    8. Tracking-loss safety
    9. Multiple-person priority
    10. Multiple-person following
"""


from robot_interface import (
    SimulatedRobot,
    RobotState
)

from decision_engine import (
    DecisionEngine,
    DecisionConfig
)


# ==========================================================
# TEST HELPERS
# ==========================================================

def make_person(
    pid,
    bbox,
    landmarks=None,
    action="unknown",
    confidence=1.0
):
    """
    Create a synthetic tracked person.

    This mimics the dictionary produced by the real
    YOLO + Tracker + MediaPipe + VideoMAE pipeline.
    """

    return {
        "id": pid,
        "bbox": bbox,
        "landmarks": landmarks,
        "action": action,
        "action_confidence": confidence
    }


# ==========================================================
# LANDMARK GENERATOR
# ==========================================================

def make_landmarks(overrides=None):
    """
    Build a basic standing pose.

    Overrides can replace individual landmark positions.

    Example:

        make_landmarks({
            "LEFT_SHOULDER": (200, 300)
        })
    """

    base = {

        "LEFT_SHOULDER":
            (300, 200),

        "RIGHT_SHOULDER":
            (340, 200),

        "LEFT_HIP":
            (305, 350),

        "RIGHT_HIP":
            (335, 350),

        "LEFT_WRIST":
            (280, 320),

        "RIGHT_WRIST":
            (360, 320),

    }


    if overrides:
        base.update(overrides)


    return [

        {
            "name": name,

            "x": xy[0],

            "y": xy[1],

            "z": 0.0,

            "visibility": 0.9

        }

        for name, xy in base.items()

    ]


# ==========================================================
# FRAME RUNNER
# ==========================================================

def run_frames(
    engine,
    actions_over_time,
    pid=1,
    bbox=(250, 150, 400, 400),
    landmarks=None
):
    """
    Feed multiple frames into the DecisionEngine.

    Each frame contains one tracked person.

    Returns:
        Last decision.
    """

    last_decision = None


    if landmarks is None:

        landmarks = make_landmarks({})


    for i, action in enumerate(
        actions_over_time
    ):

        people = [

            make_person(

                pid,

                bbox,

                landmarks=landmarks,

                action=action,

                confidence=1.0

            )

        ]


        last_decision = engine.step(
            i,
            people
        )


    return last_decision


# ==========================================================
# TEST 1
# ==========================================================

def test_idle_when_no_one_present():

    robot = SimulatedRobot()

    engine = DecisionEngine(
        robot,
        frame_width=640
    )


    decision = engine.step(
        0,
        []
    )


    assert (
        decision["decision"]
        == "idle"
    )


    assert (
        robot.status.state
        == RobotState.IDLE
    )


# ==========================================================
# TEST 2
# ==========================================================

def test_follows_walking_person_after_debounce():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=3
        )

    )


    decision = run_frames(

        engine,

        [
            "walking",
            "walking",
            "walking",
            "walking",
            "walking"
        ]

    )


    assert (
        decision["decision"]
        == "follow"
    )


    assert (
        robot.status.state
        == RobotState.FOLLOWING
    )


# ==========================================================
# TEST 3
# ==========================================================

def test_approaches_on_waving():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=3
        )

    )


    decision = run_frames(

        engine,

        [
            "waving",
            "waving",
            "waving",
            "waving",
            "waving"
        ]

    )


    assert (
        decision["decision"]
        == "approach_assist"
    )


    assert (
        robot.status.state
        == RobotState.APPROACHING
    )


# ==========================================================
# TEST 4
# ==========================================================

def test_emergency_stop_on_fall_geometry():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640

    )


    # Torso roughly horizontal.
    #
    # This should trigger the independent
    # MediaPipe pose-based fall detector.

    fallen_landmarks = make_landmarks({

        "LEFT_SHOULDER":
            (200, 300),

        "RIGHT_SHOULDER":
            (200, 340),

        "LEFT_HIP":
            (400, 305),

        "RIGHT_HIP":
            (400, 335),

    })


    people = [

        make_person(

            1,

            (150, 280, 420, 350),

            landmarks=fallen_landmarks,

            # Deliberately incorrect action.
            action="walking"

        )

    ]


    decision = engine.step(
        0,
        people
    )


    assert (
        decision["decision"]
        == "emergency_stop"
    )


    assert (
        decision["reason"]
        == "pose_fall"
    )


    assert (
        robot.status.state
        == RobotState.EMERGENCY_STOP
    )


    assert robot.in_emergency


# ==========================================================
# TEST 5
# ==========================================================

def test_emergency_stop_on_videomae_falling():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640

    )


    # Normal standing landmarks.
    #
    # We intentionally tell the engine that VideoMAE
    # detected "falling".
    #
    # This verifies the second safety path.

    people = [

        make_person(

            1,

            (250, 150, 400, 400),

            landmarks=make_landmarks({}),

            action="falling",

            confidence=0.92

        )

    ]


    decision = engine.step(
        0,
        people
    )


    assert (
        decision["decision"]
        == "emergency_stop"
    )


    assert (
        decision["reason"]
        == "action_fall"
    )


    assert (
        decision["detail"]["id"]
        == 1
    )


    assert (
        decision["detail"]["confidence"]
        == 0.92
    )


    assert (
        robot.status.state
        == RobotState.EMERGENCY_STOP
    )


    assert robot.in_emergency


# ==========================================================
# TEST 6
# ==========================================================

def test_pose_fall_has_priority_over_walking():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640

    )


    fallen_landmarks = make_landmarks({

        "LEFT_SHOULDER":
            (200, 300),

        "RIGHT_SHOULDER":
            (200, 340),

        "LEFT_HIP":
            (400, 305),

        "RIGHT_HIP":
            (400, 335),

    })


    people = [

        make_person(

            1,

            (150, 280, 420, 350),

            landmarks=fallen_landmarks,

            action="walking",

            confidence=0.95

        )

    ]


    decision = engine.step(
        0,
        people
    )


    # Safety must override walking.

    assert (
        decision["decision"]
        == "emergency_stop"
    )


    assert (
        robot.status.state
        == RobotState.EMERGENCY_STOP
    )


# ==========================================================
# TEST 7
# ==========================================================

def test_emergency_latches_until_cleared():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640

    )


    # Trigger emergency manually.

    robot.emergency_stop(
        reason="test trigger"
    )


    decision = engine.step(

        0,

        [

            make_person(

                1,

                (250, 150, 400, 400),

                action="walking"

            )

        ]

    )


    assert (
        decision["decision"]
        == "holding_emergency_stop"
    )


    assert robot.in_emergency


    # Clear manually.

    robot.clear_emergency()


    assert not robot.in_emergency


# ==========================================================
# TEST 8
# ==========================================================

def test_proximity_triggers_emergency_stop():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            proximity_area_ratio=0.2
        )

    )


    # Huge bounding box relative to frame.

    people = [

        make_person(

            1,

            (0, 0, 620, 350),

            action="walking"

        )

    ]


    decision = engine.step(
        0,
        people
    )


    assert (
        decision["decision"]
        == "emergency_stop"
    )


    assert (
        decision["reason"]
        == "proximity_risk"
    )


    assert robot.in_emergency


# ==========================================================
# TEST 9
# ==========================================================

def test_target_lost_triggers_stop():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(

            debounce_frames=2,

            lost_tracking_grace=2

        )

    )


    # Establish Person 1 as following target.

    run_frames(

        engine,

        [
            "walking",
            "walking",
            "walking"
        ]

    )


    # Missing for one frame.
    engine.step(
        3,
        []
    )


    # Missing long enough to exceed grace.
    decision = engine.step(
        5,
        []
    )


    assert (
        decision["decision"]
        == "stop_target_lost"
    )


# ==========================================================
# TEST 10
# ==========================================================

def test_falling_person_has_priority_over_walking_person():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640

    )


    # Person 1 is walking.

    walking_person = make_person(

        1,

        (100, 150, 250, 400),

        landmarks=make_landmarks({}),

        action="walking",

        confidence=0.9

    )


    # Person 2 is falling.

    fallen_landmarks = make_landmarks({

        "LEFT_SHOULDER":
            (200, 300),

        "RIGHT_SHOULDER":
            (200, 340),

        "LEFT_HIP":
            (400, 305),

        "RIGHT_HIP":
            (400, 335),

    })


    falling_person = make_person(

        2,

        (300, 280, 600, 350),

        landmarks=fallen_landmarks,

        action="walking",

        confidence=0.9

    )


    people = [

        walking_person,

        falling_person

    ]


    decision = engine.step(
        0,
        people
    )


    # Falling person must win over following.

    assert (
        decision["decision"]
        == "emergency_stop"
    )


    assert (
        decision["reason"]
        == "pose_fall"
    )


    assert robot.in_emergency


# ==========================================================
# TEST 11
# ==========================================================

def test_waving_person_has_priority_over_walking_person():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=2
        )

    )


    # First establish both people.

    for frame_idx in range(3):

        people = [

            make_person(

                1,

                (100, 150, 250, 400),

                landmarks=make_landmarks({}),

                action="walking"

            ),

            make_person(

                2,

                (350, 150, 500, 400),

                landmarks=make_landmarks({}),

                action="waving"

            )

        ]


        decision = engine.step(
            frame_idx,
            people
        )


    # Assistance has higher priority than following.

    assert (
        decision["decision"]
        == "approach_assist"
    )


    assert (
        decision["target"]
        == 2
    )


    assert (
        robot.status.state
        == RobotState.APPROACHING
    )


# ==========================================================
# TEST 12
# ==========================================================

def test_multiple_people_have_independent_actions():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=3
        )

    )


    # Person 1 walks.
    # Person 2 stands/unknown.

    for frame_idx in range(4):

        people = [

            make_person(

                1,

                (100, 150, 250, 400),

                landmarks=make_landmarks({}),

                action="walking"

            ),

            make_person(

                2,

                (350, 150, 500, 400),

                landmarks=make_landmarks({}),

                action="unknown"

            )

        ]


        decision = engine.step(
            frame_idx,
            people
        )


    # Person 1 should become the target.

    assert (
        decision["decision"]
        == "follow"
    )


    assert (
        decision["target"]
        == 1
    )


    assert (
        robot.status.state
        == RobotState.FOLLOWING
    )


# ==========================================================
# TEST 13
# ==========================================================

def test_help_request_triggers_assistance():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=2
        )

    )


    decision = run_frames(

        engine,

        [
            "help_request",
            "help_request",
            "help_request"
        ]

    )


    assert (
        decision["decision"]
        == "approach_assist"
    )


    assert (
        decision["target"]
        == 1
    )


    assert (
        robot.status.state
        == RobotState.APPROACHING
    )


# ==========================================================
# TEST 14
# ==========================================================

def test_distress_gesture_triggers_assistance():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=2
        )

    )


    decision = run_frames(

        engine,

        [
            "distress_gesture",
            "distress_gesture",
            "distress_gesture"
        ]

    )


    assert (
        decision["decision"]
        == "approach_assist"
    )


    assert (
        decision["target"]
        == 1
    )


# ==========================================================
# TEST 15
# ==========================================================

def test_unknown_person_does_not_trigger_following():

    robot = SimulatedRobot()


    engine = DecisionEngine(

        robot,

        frame_width=640,

        config=DecisionConfig(
            debounce_frames=3
        )

    )


    decision = run_frames(

        engine,

        [
            "unknown",
            "unknown",
            "unknown",
            "unknown"
        ]

    )


    assert (
        decision["decision"]
        == "idle"
    )


    assert (
        robot.status.state
        == RobotState.IDLE
    )


# ==========================================================
# DIRECT TEST EXECUTION
# ==========================================================

if __name__ == "__main__":

    import sys
    import pytest

    sys.exit(
        pytest.main(
            [
                __file__,
                "-v"
            ]
        )
    )

