"""
tracker.py
Lightweight centroid tracker so that each detected person keeps the same ID
across frames. This is what turns "a pose per frame" into a "pose SEQUENCE
per person" (Deliverable: pose sequences), without needing a heavy
re-identification model.

CHANGE FROM ORIGINAL:

The original matcher used a single fixed-pixel `max_distance` for every
match, regardless of how big the person's bounding box was. In practice
this meant:

- A person close to the camera (large bbox) has larger natural
  frame-to-frame centroid jitter in pixels, even standing still, and
  kept exceeding the fixed threshold -> track dropped -> new ID
  registered a moment later.
- A person far from the camera (small bbox) could get matched to an
  unrelated nearby detection more easily than intended.

This version scales the match threshold to the size of the bounding
boxes involved (their diagonal), so tolerance grows for people who are
close to the camera and shrinks for people who are far away. A fixed
floor (`max_distance`) is still enforced so very small/far boxes don't
get an unreasonably tight threshold.
"""

import math


class CentroidTracker:
    def __init__(
        self,
        max_distance: int = 80,
        max_missed_frames: int = 15,
        distance_scale: float = 0.75,
    ):
        """
        max_distance: minimum pixel distance floor for a match, regardless
                      of bbox size (protects small/far-away boxes from an
                      overly tight threshold).
        max_missed_frames: how many frames a track can go undetected before
                            it's dropped (handles brief occlusion).
        distance_scale: multiplier applied to the larger bbox diagonal to
                         get the adaptive threshold. Higher = more
                         tolerant of movement/jitter, but more likely to
                         merge two different nearby people. 0.75 is a
                         reasonable starting point; tune based on your log.
        """
        self.next_id = 0
        self.tracks = {}  # id -> {"centroid": (x, y), "bbox": (x1,y1,x2,y2), "missed": int}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.distance_scale = distance_scale

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _diagonal(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return math.hypot(w, h)

    def _match_threshold(self, track_bbox, det_bbox):
        """
        Adaptive matching distance: scales with how large the bounding
        boxes are (i.e. how close the person is to the camera), with a
        fixed floor so far-away/small boxes don't get an unreasonably
        loose threshold either.
        """
        diag = max(self._diagonal(track_bbox), self._diagonal(det_bbox))
        return max(self.max_distance, diag * self.distance_scale)

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
                det["id"] = self._register(centroid, det["bbox"])
            return detections

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        # distance matrix + per-pair adaptive threshold
        # rows = existing tracks, cols = new detections
        dist_matrix = [
            [math.dist(tc, ic) for ic in input_centroids]
            for tc in track_centroids
        ]

        threshold_matrix = [
            [
                self._match_threshold(track_bboxes[i], detections[j]["bbox"])
                for j in range(len(detections))
            ]
            for i in range(len(track_ids))
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
            if dist > threshold_matrix[i][j]:
                continue
            tid = track_ids[i]
            self.tracks[tid]["centroid"] = input_centroids[j]
            self.tracks[tid]["bbox"] = detections[j]["bbox"]
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
                det["id"] = self._register(input_centroids[j], det["bbox"])

        return detections

    def _register(self, centroid, bbox): 
        tid = self.next_id
        self.tracks[tid] = {"centroid": centroid, "bbox": bbox, "missed": 0}
        self.next_id += 1
        return tid