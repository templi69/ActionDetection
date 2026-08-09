"""
visualization.py
Member 2 - Deliverable: Visualization of skeletons
Draws bounding boxes, skeleton connections, and landmark points on a frame.
"""

import cv2
from pose_extraction import PoseExtractor

# A distinct color per tracked person ID (cycled), so it's easy to tell
# people apart visually in multi-person footage.
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255),
]


def _color_for_id(person_id):
    return _COLORS[person_id % len(_COLORS)]


def draw_skeleton(frame, landmarks, color=(0, 255, 0), point_radius=4, line_thickness=2,
                   visibility_threshold=0.5):
    """Draw the pose skeleton (connections + joints) for one person."""
    if not landmarks:
        return frame

    # landmarks is a list ordered the same way as PoseExtractor.LANDMARK_NAMES,
    # so index == MediaPipe landmark index, which POSE_CONNECTIONS refers to.
    points = [(int(lm["x"]), int(lm["y"]), lm["visibility"]) for lm in landmarks]

    for start_idx, end_idx in PoseExtractor.POSE_CONNECTIONS:
        x1, y1, v1 = points[start_idx]
        x2, y2, v2 = points[end_idx]
        if v1 < visibility_threshold or v2 < visibility_threshold:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), color, line_thickness)

    for x, y, v in points:
        if v < visibility_threshold:
            continue
        cv2.circle(frame, (x, y), point_radius, color, -1)

    return frame


def draw_bbox(frame, bbox, person_id=None, confidence=None, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_parts = []
    if person_id is not None:
        label_parts.append(f"ID {person_id}")
    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")
    if label_parts:
        label = " | ".join(label_parts)
        cv2.putText(frame, label, (x1, max(0, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def draw_tracked_people(frame, tracked_people):
    """
    Convenience function: draws bbox + skeleton for every tracked person
    in one call. tracked_people items look like:
        {"id": int, "bbox": (...), "confidence": float, "landmarks": [...] or None}
    """
    for person in tracked_people:
        color = _color_for_id(person["id"])
        draw_bbox(frame, person["bbox"], person["id"], person.get("confidence"), color)
        if person.get("landmarks"):
            draw_skeleton(frame, person["landmarks"], color)
    return frame
