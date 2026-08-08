from collections import defaultdict, deque


class ActionBuffer:
    """
    Maintains a temporal frame buffer for each tracked person.

    Each person gets their own queue:
        person_id -> latest N frames
    """

    def __init__(self, sequence_length=16):
        self.sequence_length = sequence_length
        self.buffers = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )

    def add_frame(self, person_id, frame):
        """
        Add the current frame for a tracked person.

        Args:
            person_id: Unique tracker ID.
            frame: OpenCV BGR frame.

        Returns:
            list of frames if buffer is full,
            otherwise None.
        """

        self.buffers[person_id].append(frame)

        if len(self.buffers[person_id]) == self.sequence_length:
            return list(self.buffers[person_id])

        return None

    def get_sequence(self, person_id):
        """Return the current sequence for a person."""
        if person_id not in self.buffers:
            return None

        if len(self.buffers[person_id]) < self.sequence_length:
            return None

        return list(self.buffers[person_id])

    def remove_person(self, person_id):
        """Remove a person's buffer when tracking is lost."""
        if person_id in self.buffers:
            del self.buffers[person_id]

    def clear(self):
        """Clear all person buffers."""
        self.buffers.clear()

    def __len__(self):
        return len(self.buffers)