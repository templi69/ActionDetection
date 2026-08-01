"""
detection.py
Member 2 - Deliverable: Detection Module
Wraps a YOLO model (Ultralytics) to detect humans in a frame.
"""

from ultralytics import YOLO


class PersonDetector:
    """Detects people in a video frame using YOLO (COCO class 0 = 'person')."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        model_path: path or name of a YOLO weights file (yolov8n.pt is small/fast,
                    good for a laptop/CPU demo; swap for yolov8m.pt/yolov8l.pt for accuracy)
        conf_threshold: minimum confidence to keep a detection
        device: 'cpu', 'cuda', or 'cuda:0' etc.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame):
        """
        Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns a list of dicts:
            {"bbox": (x1, y1, x2, y2), "confidence": float}
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                })
        return detections

    @staticmethod
    def crop_person(frame, bbox, padding: int = 20):
        """
        Crop the detected person out of the frame with a bit of padding
        (helps MediaPipe see the full body, especially near the edges of the box).

        Returns (crop, offset) where offset = (x1, y1) of the crop in the
        original frame, needed to map pose landmarks back to full-frame coords.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1)
