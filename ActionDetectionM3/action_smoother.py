from collections import Counter, deque


class ActionSmoother:
    """
    Stabilizes VideoMAE predictions using a temporal voting window.
    """

    def __init__(
        self,
        window_size=7,
        confidence_threshold=0.15
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        self.predictions = deque(
            maxlen=window_size
        )

        self.current_action = "unknown"
        self.current_confidence = 0.0

    def update(self, action, confidence):
        """
        Add a new VideoMAE prediction.

        Returns:
            stable action and confidence.
        """

        # Ignore very weak predictions
        if confidence < self.confidence_threshold:
            return {
                "action": self.current_action,
                "confidence": self.current_confidence
            }

        self.predictions.append(
            (action, confidence)
        )

        # Count actions
        action_counts = Counter(
            prediction[0]
            for prediction in self.predictions
        )

        # Most common action
        stable_action, count = action_counts.most_common(1)[0]

        # Average confidence for that action
        action_confidences = [
            confidence
            for action, confidence in self.predictions
            if action == stable_action
        ]

        stable_confidence = sum(
            action_confidences
        ) / len(action_confidences)

        self.current_action = stable_action
        self.current_confidence = stable_confidence

        return {
            "action": stable_action,
            "confidence": stable_confidence
        }

    def reset(self):
        """Clear all previous predictions."""

        self.predictions.clear()

        self.current_action = "unknown"
        self.current_confidence = 0.0