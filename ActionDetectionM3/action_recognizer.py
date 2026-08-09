import atexit
import json
from collections import defaultdict

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

    DIAGNOSTIC LOGGING:

    Every raw prediction (before filtering) is tallied by label,
    with a running count and average confidence. On process exit,
    this is written to `raw_action_log.json`, sorted by frequency.

    Use this to see which real Kinetics labels actually come up
    during normal use (walking, waving, sitting idle, etc.) instead
    of guessing label strings from memory when building ACTION_MAP.
    """

    # ==========================================================
    # ROBOT-RELEVANT ACTIONS
    #
    # NOTE: verify these against your actual model's labels using
    # list_kinetics_labels.py -- don't trust this list blindly.
    # ==========================================================

    ACTION_MAP = {

        # ------------------------------------------------------
        # Following
        # ------------------------------------------------------

        "walking":
            "walking",

        "walking the dog":
            "walking",

        "walking through snow":
            "walking",

        "walking on treadmill":
            "walking",

        "jogging":
            "walking",

        "marching":
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
        sequence_length=16,
        log_path="raw_action_log.json",
    ):

        self.sequence_length = sequence_length
        self.log_path = log_path

        # person_id-agnostic tally: label -> {"count": int, "total_confidence": float}
        self._raw_label_stats = defaultdict(
            lambda: {"count": 0, "total_confidence": 0.0}
        )

        # Write the log automatically when the process exits
        # (Ctrl+C, normal exit, etc.) so you don't lose it.
        atexit.register(self._write_log)

        # ======================================================
        # DEVICE
        # ======================================================

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # ======================================================
        # PROCESSOR
        # ======================================================

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        # ======================================================
        # MODEL
        # ======================================================

        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("VideoMAE loaded successfully.")
        print(f"Number of classes: {self.model.config.num_labels}")
        print("Robot action filtering: ENABLED")
        print(f"Raw label diagnostics will be written to: {self.log_path}")

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

        if len(frames) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} frames, "
                f"but received {len(frames)}"
            )

        # BGR -> RGB
        rgb_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]

        inputs = self.processor(rgb_frames, return_tensors="pt")

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        confidence, class_id = torch.max(probabilities, dim=-1)

        confidence = confidence.item()
        class_id = class_id.item()

        raw_action = self.model.config.id2label[class_id]
        raw_action = raw_action.strip().lower()

        # --------------------------------------------------
        # DIAGNOSTIC TALLY
        # --------------------------------------------------

        stats = self._raw_label_stats[raw_action]
        stats["count"] += 1
        stats["total_confidence"] += confidence

        normalized_action = self.ACTION_MAP.get(raw_action, "unknown")

        return {
            "action": normalized_action,
            "confidence": confidence,
            "class_id": class_id,
            "raw_action": raw_action,
        }

    # ==========================================================
    # DIAGNOSTIC LOG
    # ==========================================================

    def _write_log(self):

        if not self._raw_label_stats:
            return

        report = []

        for label, stats in self._raw_label_stats.items():
            count = stats["count"]
            avg_confidence = stats["total_confidence"] / count
            report.append({
                "raw_label": label,
                "count": count,
                "avg_confidence": round(avg_confidence, 3),
                "mapped_to": self.ACTION_MAP.get(label, "unknown"),
            })

        # Most frequent labels first.
        report.sort(key=lambda r: r["count"], reverse=True)

        try:
            with open(self.log_path, "w") as f:
                json.dump(report, f, indent=2)

            print()
            print(f"[VideoMAE] Wrote raw label diagnostics to {self.log_path}")
            print("[VideoMAE] Top raw labels seen this run:")

            for row in report[:10]:
                print(
                    f"  {row['raw_label']:<30} "
                    f"count={row['count']:<5} "
                    f"avg_conf={row['avg_confidence']:<6} "
                    f"-> {row['mapped_to']}"
                )

        except Exception as e:
            print(f"[VideoMAE] Failed to write raw action log: {e}")