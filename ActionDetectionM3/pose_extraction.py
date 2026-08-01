"""
pose_extraction.py
Member 2 - Deliverable: Pose Extraction Module
Wraps MediaPipe Pose to extract 33 body landmarks from a cropped person image.
"""

import cv2
import mediapipe as mp


class PoseExtractor:
    """Extracts body landmarks from a person crop using MediaPipe Pose."""

    LANDMARK_NAMES = [lm.name for lm in mp.solutions.pose.PoseLandmark]
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS  # used by visualization.py

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._mp_pose = mp.solutions.pose
        self.pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, crop, offset=(0, 0)):
        """
        Run pose estimation on a cropped person image.

        crop: BGR image (output of PersonDetector.crop_person)
        offset: (x, y) top-left corner of this crop in the original frame,
                so landmarks can be reported in full-frame pixel coordinates.

        Returns a list of 33 landmark dicts, or None if no pose was found:
            {"name": str, "x": float, "y": float, "z": float, "visibility": float}
        x, y are in ORIGINAL FRAME pixel coordinates.
        z is MediaPipe's relative depth (roughly same scale as x, hip-centered).
        visibility is 0-1 confidence that the landmark is visible/not occluded.
        """
        if crop is None or crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = crop.shape[:2]
        ox, oy = offset

        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "name": self.LANDMARK_NAMES[idx],
                "x": lm.x * w + ox,
                "y": lm.y * h + oy,
                "z": lm.z,
                "visibility": lm.visibility,
            })
        return landmarks

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
