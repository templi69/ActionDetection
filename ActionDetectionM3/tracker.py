"""
tracker.py
Lightweight centroid tracker so that each detected person keeps the same ID
across frames. This is what turns "a pose per frame" into a "pose SEQUENCE
per person" (Deliverable: pose sequences), without needing a heavy
re-identification model.
"""

import math


class CentroidTracker:
    def __init__(self, max_distance: int = 80, max_missed_frames: int = 15):
        """
        max_distance: max pixel distance between frames to consider it the
                      same person (tune based on frame resolution / FPS)
        max_missed_frames: how many frames a track can go undetected before
                            it's dropped (handles brief occlusion)
        """
        self.next_id = 0
        self.tracks = {}  # id -> {"centroid": (x, y), "missed": int}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections):
        """
        detections: list of {"bbox": (x1,y1,x2,y2), "confidence": float}

        Returns the same list with an added "id" key per detection, and
        internally ages out tracks that were not matched.
        """
        input_centroids = [self._centroid(d["bbox"]) for d in detections]

        # No existing tracks yet: register everything as new
        if not self.tracks:
            for det, centroid in zip(detections, input_centroids):
                det["id"] = self._register(centroid)
            return detections

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]

        # distance matrix: rows = existing tracks, cols = new detections
        dist_matrix = [
            [math.dist(tc, ic) for ic in input_centroids]
            for tc in track_centroids
        ]

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy nearest-neighbor matching (fine for a handful of people)
        pairs = []
        for i, row in enumerate(dist_matrix):
            for j, dist in enumerate(row):
                pairs.append((dist, i, j))
        pairs.sort(key=lambda p: p[0])

        for dist, i, j in pairs:
            if i in assigned_tracks or j in assigned_dets:
                continue
            if dist > self.max_distance:
                continue
            tid = track_ids[i]
            self.tracks[tid]["centroid"] = input_centroids[j]
            self.tracks[tid]["missed"] = 0
            detections[j]["id"] = tid
            assigned_tracks.add(i)
            assigned_dets.add(j)

        # Unmatched existing tracks -> age them
        for i, tid in enumerate(track_ids):
            if i not in assigned_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed_frames:
                    del self.tracks[tid]

        # Unmatched detections -> new tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                det["id"] = self._register(input_centroids[j])

        return detections

    def _register(self, centroid):
        tid = self.next_id
        self.tracks[tid] = {"centroid": centroid, "missed": 0}
        self.next_id += 1
        return tid
