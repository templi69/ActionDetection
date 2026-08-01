"""
sequence.py
Member 2 - Deliverable: Generate pose sequences
Collects per-frame, per-person landmark data into sequences and exports
them to JSON for downstream use (e.g. action recognition, animation, VR).
"""

import json
from collections import defaultdict


class PoseSequenceBuilder:
    """Accumulates landmarks per tracked person ID, frame by frame."""

    def __init__(self):
        # person_id -> list of {"frame": int, "landmarks": [...]}
        self.sequences = defaultdict(list)

    def add_frame(self, frame_idx, tracked_people):
        """
        tracked_people: list of dicts like
            {"id": int, "bbox": (...), "confidence": float, "landmarks": [...] or None}
        """
        for person in tracked_people:
            if person.get("landmarks") is None:
                continue
            self.sequences[person["id"]].append({
                "frame": frame_idx,
                "bbox": person["bbox"],
                "landmarks": person["landmarks"],
            })

    def get_sequence(self, person_id):
        return self.sequences.get(person_id, [])

    def num_people(self):
        return len(self.sequences)

    def to_dict(self):
        return {str(pid): frames for pid, frames in self.sequences.items()}

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
