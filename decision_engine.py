"""
decision_engine.py
Member 4 - Deliverable: Decision Logic
Maps recognized person actions + safety signals to robot commands.

Expected upstream input per tracked person, per frame (from Member 3's
action recognition, layered on top of Member 2's pose sequences):

    {
        "id": int,
        "bbox": (x1, y1, x2, y2),
        "landmarks": [...] or None,
        "action": str,          # e.g. "walking", "waving", "falling", "idle", "unknown"
        "action_confidence": float,
    }

If Member 3's module isn't ready yet, use safety.infer_fallback_action()
to derive a rough action straight from landmarks so this module can be
developed/tested independently (see robot_pipeline.py for an example).
"""

from collections import deque, Counter
from dataclasses import dataclass

from robot_interface import RobotState
from safety import check_fall, check_proximity_risk, check_tracking_lost


@dataclass
class DecisionConfig:
    # how many consecutive frames an action must persist before acting on it
    # (debounces flicker from a noisy per-frame classifier)
    debounce_frames: int = 5
    # frame area fraction above which we treat a bbox as "too close"
    proximity_area_ratio: float = 0.35
    # frames a previously-tracked target can vanish before we call it "lost"
    lost_tracking_grace: int = 20
    follow_linear_speed: float = 0.4
    approach_linear_speed: float = 0.25
    turn_gain: float = 0.003  # rad/s per pixel of horizontal offset from center


class DecisionEngine:
    """
    Priority order every frame (highest first):
      1. Emergency stop conditions (fall, collision risk) -> latch stop
      2. Assistance requests (waving / distress gesture) -> approach + alert
      3. Following behavior (walking target) -> follow
      4. Idle (nobody actionable) -> hold position
    """

    def __init__(self, robot, frame_width: int, config: DecisionConfig = None):
        self.robot = robot
        self.frame_width = frame_width
        self.cfg = config or DecisionConfig()
        self._action_history = {}      # person_id -> deque of recent action labels
        self._last_seen_frame = {}     # person_id -> frame_idx last seen
        self._primary_target_id = None

    def step(self, frame_idx, tracked_people):
        """Call once per frame. Returns a dict describing the decision made (for logging/UI)."""

        # ---- 1. Emergency conditions take priority over everything ----
        fallen = check_fall(tracked_people)
        if fallen:
            self.robot.emergency_stop(reason=f"person {fallen['id']} appears to have fallen")
            return {"decision": "emergency_stop", "detail": fallen}

        collision_risk = check_proximity_risk(tracked_people, self.frame_width, self.cfg.proximity_area_ratio)
        if collision_risk:
            self.robot.emergency_stop(reason=f"person {collision_risk['id']} too close to robot")
            return {"decision": "emergency_stop", "detail": collision_risk}

        if self.robot.in_emergency:
            # Stay latched until a human/operator clears it via robot.clear_emergency().
            return {"decision": "holding_emergency_stop"}

        # ---- housekeeping for debounce + lost-track detection ----
        seen_ids = set()
        for person in tracked_people:
            pid = person["id"]
            seen_ids.add(pid)
            self._last_seen_frame[pid] = frame_idx
            history = self._action_history.setdefault(pid, deque(maxlen=self.cfg.debounce_frames))
            history.append(person.get("action", "unknown"))

        if self._primary_target_id is not None and self._primary_target_id not in seen_ids:
            lost = check_tracking_lost(
                self._primary_target_id, frame_idx, self._last_seen_frame, self.cfg.lost_tracking_grace
            )
            if lost:
                self.robot.stop(reason="tracked person lost")
                self._primary_target_id = None
                return {"decision": "stop_target_lost"}

        # ---- 2. Assistance requests ----
        for person in tracked_people:
            stable_action = self._stable_action(person["id"])
            if stable_action in ("waving", "distress_gesture", "help_request"):
                self._primary_target_id = person["id"]
                self._drive_towards(person, self.cfg.approach_linear_speed)
                self.robot.set_state(RobotState.APPROACHING)
                self.robot.alert(f"Assisting person {person['id']}")
                return {"decision": "approach_assist", "target": person["id"]}

        # ---- 3. Following ----
        for person in tracked_people:
            stable_action = self._stable_action(person["id"])
            if stable_action == "walking" and (
                self._primary_target_id is None or self._primary_target_id == person["id"]
            ):
                self._primary_target_id = person["id"]
                self._drive_towards(person, self.cfg.follow_linear_speed)
                self.robot.set_state(RobotState.FOLLOWING)
                return {"decision": "follow", "target": person["id"]}

        # ---- 4. Idle ----
        self.robot.stop(reason="no actionable person")
        self.robot.set_state(RobotState.IDLE)
        return {"decision": "idle"}

    def _stable_action(self, person_id):
        """Majority vote over the debounce window; avoids reacting to one noisy frame."""
        history = self._action_history.get(person_id)
        if not history or len(history) < self.cfg.debounce_frames:
            return "unknown"
        return Counter(history).most_common(1)[0][0]

    def _drive_towards(self, person, linear_speed):
        x1, y1, x2, y2 = person["bbox"]
        cx = (x1 + x2) / 2.0
        offset = cx - (self.frame_width / 2.0)    # + means target is to the right
        angular = -self.cfg.turn_gain * offset     # turn towards it
        self.robot.move(linear=linear_speed, angular=angular, reason=f"tracking id {person['id']}")
