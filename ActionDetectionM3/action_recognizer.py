import torch
import cv2

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)


class VideoMAEActionRecognizer:
    """
    VideoMAE action recognizer.

    VideoMAE is currently using a Kinetics-400 pretrained model.

    Since Kinetics contains many actions that are irrelevant to our
    robot, this class filters and normalizes the model output into
    actions understood by the DecisionEngine.
    """

    # ==========================================================
    # ROBOT-RELEVANT ACTIONS
    # ==========================================================

    ACTION_MAP = {

        # ------------------------------------------------------
        # Following
        # ------------------------------------------------------

        "walking":
            "walking",

        "walking through snow":
            "walking",

        "walking on treadmill":
            "walking",

        # ------------------------------------------------------
        # Assistance / waving
        # ------------------------------------------------------

        "waving hand":
            "waving",

        "waving":
            "waving",

        # ------------------------------------------------------
        # Falling
        # ------------------------------------------------------

        "falling down":
            "falling",

        "falling":
            "falling",

        # ------------------------------------------------------
        # Idle / stationary
        # ------------------------------------------------------

        "standing":
            "idle",

        "sitting":
            "idle",

        "sitting down":
            "idle",
    }


    def __init__(
        self,
        model_name="MCG-NJU/videomae-base-finetuned-kinetics",
        device=None,
        sequence_length=16
    ):

        self.sequence_length = sequence_length


        # ======================================================
        # DEVICE
        # ======================================================

        if device is None:

            self.device = torch.device(

                "cuda"
                if torch.cuda.is_available()
                else "cpu"

            )

        else:

            self.device = torch.device(
                device
            )


        print(
            f"Using device: {self.device}"
        )


        # ======================================================
        # PROCESSOR
        # ======================================================

        self.processor = (
            VideoMAEImageProcessor
            .from_pretrained(
                model_name
            )
        )


        # ======================================================
        # MODEL
        # ======================================================

        self.model = (
            VideoMAEForVideoClassification
            .from_pretrained(
                model_name
            )
        )


        self.model.to(
            self.device
        )


        self.model.eval()


        print(
            "VideoMAE loaded successfully."
        )


        # ======================================================
        # SHOW MODEL INFORMATION
        # ======================================================

        print(
            f"Number of classes: "
            f"{self.model.config.num_labels}"
        )


        print(
            "Robot action filtering: ENABLED"
        )


    # ==========================================================
    # PREDICTION
    # ==========================================================

    @torch.no_grad()
    def predict(self, frames):
        """
        Predict an action from exactly sequence_length frames.

        Returns:

        {
            "action": str,
            "confidence": float,
            "class_id": int,
            "raw_action": str
        }
        """

        # ======================================================
        # VALIDATE FRAME COUNT
        # ======================================================

        if len(frames) != self.sequence_length:

            raise ValueError(

                f"Expected "
                f"{self.sequence_length} frames, "

                f"but received "
                f"{len(frames)}"

            )


        # ======================================================
        # BGR -> RGB
        # ======================================================

        rgb_frames = [

            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            for frame in frames

        ]


        # ======================================================
        # PROCESS VIDEO
        # ======================================================

        inputs = self.processor(

            rgb_frames,

            return_tensors="pt"

        )


        # ======================================================
        # MOVE TO DEVICE
        # ======================================================

        inputs = {

            key: value.to(
                self.device
            )

            for key, value in inputs.items()

        }


        # ======================================================
        # MODEL INFERENCE
        # ======================================================

        outputs = self.model(
            **inputs
        )


        # ======================================================
        # PROBABILITIES
        # ======================================================

        probabilities = torch.softmax(

            outputs.logits,

            dim=-1

        )


        confidence, class_id = (
            torch.max(
                probabilities,
                dim=-1
            )
        )


        confidence = (
            confidence.item()
        )


        class_id = (
            class_id.item()
        )


        # ======================================================
        # RAW KINETICS LABEL
        # ======================================================

        raw_action = (
            self.model.config.id2label[
                class_id
            ]
        )


        raw_action = (
            raw_action
            .strip()
            .lower()
        )


        # ======================================================
        # NORMALIZE ROBOT ACTION
        # ======================================================

        normalized_action = (
            self.ACTION_MAP.get(

                raw_action,

                "unknown"

            )
        )


        # ======================================================
        # RESULT
        # ======================================================

        return {

            "action":
                normalized_action,

            "confidence":
                confidence,

            "class_id":
                class_id,

            "raw_action":
                raw_action

        }