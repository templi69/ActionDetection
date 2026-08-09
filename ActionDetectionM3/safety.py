"""
safety.py
Member 4 - Safety Layer

Safety checks for the robot decision system.

Priority:
    1. Pose-based fall detection
    2. VideoMAE falling detection
    3. Proximity / collision risk
    4. Tracking-loss handling

VideoMAE action predictions are only considered for safety when
their confidence is >= ACTION_FALL_CONFIDENCE_THRESHOLD.
"""

import math


# ==========================================================
# CONFIGURATION
# ==========================================================

FALL_TORSO_ANGLE_DEG = 55

FALL_MIN_VISIBILITY = 0.5

# VideoMAE "falling" must be at least this confident
# before it can trigger an emergency stop.
ACTION_FALL_CONFIDENCE_THRESHOLD = 0.60


# ==========================================================
# LANDMARK HELPERS
# ==========================================================

def _get_landmark(landmarks, name):
    """
    Safely find a landmark by name.

    Returns:
        Landmark dictionary or None.
    """

    for lm in landmarks or []:

        if not isinstance(lm, dict):
            continue

        if lm.get("name") == name:
            return lm

    return None


# ==========================================================
# TORSO FALL DETECTION
# ==========================================================

def _torso_angle_from_vertical(landmarks):
    """
    Estimate torso angle relative to vertical.

    Standing person:
        approximately 0-20 degrees

    Horizontal / fallen person:
        approximately 90 degrees

    Returns:
        Angle in degrees, or None if landmarks are invalid.
    """

    ls = _get_landmark(
        landmarks,
        "LEFT_SHOULDER"
    )

    rs = _get_landmark(
        landmarks,
        "RIGHT_SHOULDER"
    )

    lh = _get_landmark(
        landmarks,
        "LEFT_HIP"
    )

    rh = _get_landmark(
        landmarks,
        "RIGHT_HIP"
    )


    if not all([ls, rs, lh, rh]):
        return None


    try:

        visibility_values = [

            float(ls.get("visibility", 0.0)),

            float(rs.get("visibility", 0.0)),

            float(lh.get("visibility", 0.0)),

            float(rh.get("visibility", 0.0))

        ]

    except (
        TypeError,
        ValueError
    ):

        return None


    if min(visibility_values) < FALL_MIN_VISIBILITY:
        return None


    try:

        shoulder_mid = (

            (
                float(ls["x"])
                +
                float(rs["x"])
            ) / 2.0,

            (
                float(ls["y"])
                +
                float(rs["y"])
            ) / 2.0

        )


        hip_mid = (

            (
                float(lh["x"])
                +
                float(rh["x"])
            ) / 2.0,

            (
                float(lh["y"])
                +
                float(rh["y"])
            ) / 2.0

        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return None


    dx = hip_mid[0] - shoulder_mid[0]

    dy = hip_mid[1] - shoulder_mid[1]


    if dx == 0 and dy == 0:
        return None


    angle = math.degrees(
        math.atan2(
            abs(dx),
            abs(dy)
        )
    )


    return angle


# ==========================================================
# POSE-BASED FALL CHECK
# ==========================================================

def check_fall(tracked_people):
    """
    Detect a fall using body geometry.

    This check is independent of VideoMAE.

    Returns:
        {
            "id": person_id,
            "torso_angle_deg": angle
        }

        or None.
    """

    for person in tracked_people:

        angle = _torso_angle_from_vertical(
            person.get("landmarks")
        )


        if (
            angle is not None
            and angle > FALL_TORSO_ANGLE_DEG
        ):

            return {

                "id": person["id"],

                "torso_angle_deg": angle

            }


    return None


# ==========================================================
# ACTION-BASED FALL CHECK
# ==========================================================

def check_action_fall(
    tracked_people,
    confidence_threshold=ACTION_FALL_CONFIDENCE_THRESHOLD
):
    """
    Detect a fall using the action recognizer.

    VideoMAE must report:

        action == "falling"

    AND:

        confidence >= 0.60

    Returns:

        {
            "id": person_id,
            "action": "falling",
            "confidence": confidence
        }

        or None.
    """

    for person in tracked_people:

        action = str(
            person.get(
                "action",
                "unknown"
            )
        ).strip().lower()


        try:

            confidence = float(
                person.get(
                    "action_confidence",
                    0.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0


        if (

            action == "falling"

            and confidence >= confidence_threshold

        ):

            return {

                "id": person["id"],

                "action": action,

                "confidence": confidence

            }


    return None


# ==========================================================
# PROXIMITY / COLLISION CHECK
# ==========================================================

def check_proximity_risk(
    tracked_people,
    frame_width,
    area_ratio_threshold,
    frame_height=None
):
    """
    Detect a person who is very close to the camera/robot.

    A large bounding-box area means the person is likely
    physically close.

    Returns:

        {
            "id": person_id,
            "area_ratio": ratio
        }

        or None.
    """

    if frame_width <= 0:
        return None


    # If actual frame height isn't supplied,
    # assume 16:9.
    if frame_height is None:

        frame_height = (
            frame_width * 0.5625
        )


    frame_area = (
        frame_width * frame_height
    )


    if frame_area <= 0:
        return None


    for person in tracked_people:

        bbox = person.get("bbox")


        if not bbox or len(bbox) != 4:
            continue


        try:

            x1, y1, x2, y2 = map(
                float,
                bbox
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        box_width = max(
            0.0,
            x2 - x1
        )

        box_height = max(
            0.0,
            y2 - y1
        )


        box_area = (
            box_width
            *
            box_height
        )


        area_ratio = (
            box_area
            /
            frame_area
        )


        if area_ratio > area_ratio_threshold:

            return {

                "id": person["id"],

                "area_ratio": area_ratio

            }


    return None


# ==========================================================
# TRACKING LOST
# ==========================================================

def check_tracking_lost(
    target_id,
    frame_idx,
    last_seen_frame,
    grace_frames
):
    """
    Determine whether a previously tracked target
    has been missing for too long.
    """

    last_seen = last_seen_frame.get(
        target_id
    )


    if last_seen is None:
        return True


    return (
        frame_idx - last_seen
    ) > grace_frames


# ==========================================================
# FALLBACK ACTION
# ==========================================================

def infer_fallback_action(landmarks):
    """
    Rough fallback action inference.

    This is NOT the main action recognizer.

    It exists as a backup when VideoMAE is unavailable.

    Possible outputs:

        falling
        waving
        walking
        unknown
    """

    if not landmarks:
        return "unknown"


    # ------------------------------------------------------
    # FALL
    # ------------------------------------------------------

    angle = _torso_angle_from_vertical(
        landmarks
    )


    if (

        angle is not None

        and angle > FALL_TORSO_ANGLE_DEG

    ):

        return "falling"


    # ------------------------------------------------------
    # WAVING
    # ------------------------------------------------------

    l_wrist = _get_landmark(
        landmarks,
        "LEFT_WRIST"
    )

    l_shoulder = _get_landmark(
        landmarks,
        "LEFT_SHOULDER"
    )

    r_wrist = _get_landmark(
        landmarks,
        "RIGHT_WRIST"
    )

    r_shoulder = _get_landmark(
        landmarks,
        "RIGHT_SHOULDER"
    )


    if l_wrist and l_shoulder:

        try:

            if (
                float(l_wrist["y"])
                <
                float(l_shoulder["y"]) - 20
            ):

                return "waving"

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            pass


    if r_wrist and r_shoulder:

        try:

            if (
                float(r_wrist["y"])
                <
                float(r_shoulder["y"]) - 20
            ):

                return "waving"

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            pass


    # ------------------------------------------------------
    # DEFAULT
    # ------------------------------------------------------

    return "walking"