import torch
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


class VideoMAEActionRecognizer:

    def __init__(
        self,
        model_name="MCG-NJU/videomae-base-finetuned-kinetics",
        device=None,
        sequence_length=16
    ):
        self.sequence_length = sequence_length

        # Select GPU automatically
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load processor
        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name
        )

        # Load VideoMAE
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print("VideoMAE loaded successfully.")

    @torch.no_grad()
    def predict(self, frames):
        """
        Predict an action from a sequence of frames.

        Args:
            frames:
                List of OpenCV BGR frames.

        Returns:
            {
                "action": str,
                "confidence": float,
                "class_id": int
            }
        """

        if len(frames) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} frames, "
                f"but received {len(frames)}"
            )

        # Convert OpenCV BGR → RGB
        rgb_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]

        # VideoMAE processor
        inputs = self.processor(
            rgb_frames,
            return_tensors="pt"
        )

        # Move tensors to GPU
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Inference
        outputs = self.model(**inputs)

        # Probabilities
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        confidence, class_id = torch.max(
            probabilities,
            dim=-1
        )

        confidence = confidence.item()
        class_id = class_id.item()

        # Convert class ID → label
        action = self.model.config.id2label[class_id]

        return {
            "action": action,
            "confidence": confidence,
            "class_id": class_id
        }