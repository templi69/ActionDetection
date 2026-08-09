from collections import Counter, deque


class ActionSmoother:
    """
    Stabilizes VideoMAE predictions using a temporal voting window.

    IMPORTANT CHANGE from the original version:

    The original smoother discarded any prediction below
    `confidence_threshold` *before* it ever entered the voting
    window. Because VideoMAE's Kinetics-400 head spreads probability
    across 400 classes, a lot of genuine predictions (waving,
    falling, etc.) never clear a per-frame 0.15 bar -- so they were
    silently dropped and the smoother stayed stuck on "unknown"
    forever.

    This version separates two different jobs that were previously
    conflated into one gate:

    1. `min_frame_confidence` -- a very low bar that only filters
       out near-zero-confidence noise. Almost everything real
       should clear this and make it into the voting window.

    2. `stable_confidence_threshold` -- applied to the *aggregated*
       (averaged, majority-voted) confidence, not the raw per-frame
       confidence. Only once several consistent frames accumulate
       enough combined confidence does the reported action actually
       change. This still filters noise, but noise gets outvoted
       over the window instead of being thrown away one frame at a
       time.
    """

    def __init__(
        self,
        window_size=7,
        min_frame_confidence=0.03,
        stable_confidence_threshold=0.15,
    ):
        self.window_size = window_size

        # Per-frame noise floor -- keep this low.
        self.min_frame_confidence = min_frame_confidence

        # Bar for the *aggregated* confidence before we accept a
        # new stable action. This is what "confidence_threshold"
        # used to mean, but now it's applied after voting instead
        # of before.
        self.stable_confidence_threshold = stable_confidence_threshold

        self.predictions = deque(maxlen=window_size)

        self.current_action = "unknown"
        self.current_confidence = 0.0

    def update(self, action, confidence):
        """
        Add a new VideoMAE prediction.

        Returns:
            stable action and confidence.
        """

        # Only reject near-zero-confidence noise. Everything else
        # gets a vote, even if it's individually weak.
        if confidence < self.min_frame_confidence:
            return {
                "action": self.current_action,
                "confidence": self.current_confidence,
            }

        self.predictions.append((action, confidence))

        # Count actions across the window.
        action_counts = Counter(
            prediction[0] for prediction in self.predictions
        )

        # Most common action in the window.
        candidate_action, count = action_counts.most_common(1)[0]

        # Average confidence for that action across the window.
        candidate_confidences = [
            conf
            for act, conf in self.predictions
            if act == candidate_action
        ]

        candidate_confidence = (
            sum(candidate_confidences) / len(candidate_confidences)
        )

        # Only accept the new stable action once the AGGREGATE
        # confidence clears the bar -- individual weak frames are
        # allowed to contribute votes without being disqualified
        # on their own.
        if candidate_confidence >= self.stable_confidence_threshold:
            self.current_action = candidate_action
            self.current_confidence = candidate_confidence

        return {
            "action": self.current_action,
            "confidence": self.current_confidence,
        }

    def reset(self):
        """Clear all previous predictions."""

        self.predictions.clear()

        self.current_action = "unknown"
        self.current_confidence = 0.0