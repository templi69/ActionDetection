"""
safety.py
Member 4 - Deliverable: Emergency stop & assistance safety checks
Pure functions so they're easy to unit test without a robot or camera.
All checks work directly off pose landmarks / bboxes, independent of
whatever action-recognition labels Member 3 provides -- this is the
safety net that still works even if that classifier is wrong or absent.
"""

import math

FALL_TORSO_ANGLE_DEG = 55          # from vertical; more horizontal => likely fallen
FALL_MIN_VISIBILITY = 0.5


def _get_landmark(landmarks, name):
    for lm in landmarks or []:
        if lm["name"] == name:
            return lm
    return None


def _torso_angle_from_vertical(landmarks):
    """
    Rough fall heuristic: angle of the line from mid-shoulder to mid-hip,
    measured from vertical. A standing/sitting person is close to 0-20deg;
    someone lying on the ground is closer to 90deg.
    """
    ls = _get_landmark(landmarks, "LEFT_SHOULDER")
    rs = _get_landmark(landmarks, "RIGHT_SHOULDER")
    lh = _get_landmark(landmarks, "LEFT_HIP")
    rh = _get_landmark(landmarks, "RIGHT_HIP")
    if not all([ls, rs, lh, rh]):
        return None
    if min(ls["visibility"], rs["visibility"], lh["visibility"], rh["visibility"]) < FALL_MIN_VISIBILITY:
        return None

    shoulder_mid = ((ls["x"] + rs["x"]) / 2.0, (ls["y"] + rs["y"]) / 2.0)
    hip_mid = ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def check_fall(tracked_people):
    """Returns the first person whose torso geometry suggests a fall, or None."""
    for person in tracked_people:
        angle = _torso_angle_from_vertical(person.get("landmarks"))
        if angle is not None and angle > FALL_TORSO_ANGLE_DEG:
            return {"id": person["id"], "torso_angle_deg": angle}
    return None


def check_proximity_risk(tracked_people, frame_width, area_ratio_threshold, frame_height=None):
    """
    Flags a person whose bbox occupies a large fraction of the frame,
    i.e. they're very close to the camera/robot -> collision risk.
    """
    frame_height = frame_height or frame_width * 0.5625  # assume 16:9 if unknown
    frame_area = frame_width * frame_height
    if frame_area <= 0:
        return None

    for person in tracked_people:
        x1, y1, x2, y2 = person["bbox"]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if (box_area / frame_area) > area_ratio_threshold:
            return {"id": person["id"], "area_ratio": box_area / frame_area}
    return None


def check_tracking_lost(target_id, frame_idx, last_seen_frame, grace_frames):
    last_seen = last_seen_frame.get(target_id)
    if last_seen is None:
        return True
    return (frame_idx - last_seen) > grace_frames


def infer_fallback_action(landmarks):
    """
    Rough placeholder action inference from landmarks alone, so this module
    (and the decision engine) can be developed and tested before Member 3's
    real action-recognition classifier exists. Replace calls to this with
    Member 3's output once it's ready -- keep this only as a dev stub /
    redundancy check, since the fall-detection path above already covers
    the safety-critical case regardless of what "action" says.
    """
    if not landmarks:
        return "unknown"

    angle = _torso_angle_from_vertical(landmarks)
    if angle is not None and angle > FALL_TORSO_ANGLE_DEG:
        return "falling"

    l_wrist = _get_landmark(landmarks, "LEFT_WRIST")
    l_shoulder = _get_landmark(landmarks, "LEFT_SHOULDER")
    r_wrist = _get_landmark(landmarks, "RIGHT_WRIST")
    r_shoulder = _get_landmark(landmarks, "RIGHT_SHOULDER")
    if l_wrist and l_shoulder and l_wrist["y"] < l_shoulder["y"] - 20:
        return "waving"
    if r_wrist and r_shoulder and r_wrist["y"] < r_shoulder["y"] - 20:
        return "waving"

    return "walking"
