"""
videomae_action.py
Member 3 - VideoMAE Action Recognition

Loads a pretrained VideoMAE model and predicts an action
from a sequence of video frames.
"""

import cv2
import torch
import numpy as np

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


class VideoMAEActionRecognizer:

    def __init__(
        self,
        model_name="MCG-NJU/videomae-base-finetuned-kinetics",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading VideoMAE on {self.device}...")

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        self.labels = self.model.config.id2label

        print("VideoMAE loaded successfully.")
        print(f"Number of classes: {len(self.labels)}")

    def predict(self, frames):
        """
        frames:
            List of BGR OpenCV frames.

        Returns:
            {
                "action": str,
                "confidence": float,
                "class_id": int
            }
        """

        if not frames:
            return {
                "action": "unknown",
                "confidence": 0.0,
                "class_id": -1,
            }

        # Convert OpenCV BGR -> RGB
        rgb_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]

        # VideoMAE normally expects a temporal clip.
        # Processor handles resizing / normalization.
        inputs = self.processor(
            list(rgb_frames),
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        confidence, class_id = torch.max(probabilities, dim=-1)

        class_id = int(class_id.item())
        confidence = float(confidence.item())

        action = self.labels[class_id]

        return {
            "action": action,
            "confidence": confidence,
            "class_id": class_id,
        }


def load_video_frames(video_path, num_frames=16):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video contains no frames.")

    # Select frames evenly throughout the video.
    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    frames = []

    for index in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        ok, frame = cap.read()

        if ok:
            frames.append(frame)

    cap.release()

    return frames


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file",
    )

    args = parser.parse_args()

    recognizer = VideoMAEActionRecognizer()

    frames = load_video_frames(
        args.video,
        num_frames=16,
    )

    print(f"Loaded {len(frames)} frames.")

    result = recognizer.predict(frames)

    print("\n========== ACTION RESULT ==========")
    print(f"Action:     {result['action']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Class ID:   {result['class_id']}")