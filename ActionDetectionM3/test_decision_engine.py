"""
decision_engine.py
Member 4 - Safety-aware decision engine.

Priority:
    1. Emergency-stop latch
    2. Pose-based fall
    3. VideoMAE falling action
    4. Proximity risk
    5. Lost target
    6. Assistance request
    7. Following
    8. Idle

Important fix:
    A lost target is checked BEFORE the generic "no people -> idle"
    path. This prevents a previously-followed person from silently
    becoming "idle" after tracking is lost.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Any

from robot_interface import RobotState


@dataclass
class DecisionConfig:
    # Action debounce
    debounce_frames: int = 3

    # Safety
    proximity_area_ratio: float = 0.35
    fall_action_confidence: float = 0.70
    lost_tracking_grace: int = 20

    # Robot motion
    follow_linear_speed: float = 0.40
    approach_linear_speed: float = 0.25
    turn_gain: float = 0.80

    # Tracking/action state
    action_history_size: int = 10


class DecisionEngine:
    """
    Converts tracked people + recognized actions into robot decisions.

    Each tracked person has an independent action history.
    """

    def __init__(
        self,
        robot,
        frame_width: int = 640,
        config: Optional[DecisionConfig] = None,
    ):
        self.robot = robot
        self.frame_width = max(1, int(frame_width))
        self.config = config or DecisionConfig()

        # person_id -> recent actions
        self._action_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.action_history_size)
        )

        # Last frame in which each person was actually visible.
        self._last_seen_frame: Dict[int, int] = {}

        # Current target being followed/approached.
        self.target_id: Optional[int] = None

        # Keep the target's last action for debugging/recovery.
        self.target_action: Optional[str] = None

    # ==========================================================
    # BASIC HELPERS
    # ==========================================================

    @staticmethod
    def _person_id(person) -> Optional[int]:
        return person.get("id")

    @staticmethod
    def _action(person) -> str:
        return str(person.get("action", "unknown")).lower().strip()

    @staticmethod
    def _confidence(person) -> float:
        try:
            return float(person.get("action_confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bbox(person):
        bbox = person.get("bbox")
        if not bbox or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = map(float, bbox)
            return x1, y1, x2, y2
        except (TypeError, ValueError):
            return None

    def _set_state(self, state):
        """
        Keep the robot state synchronized with the decision engine.

        The existing SimulatedRobot exposes robot.status.state, so this
        remains compatible with the current tests.
        """
        try:
            self.robot.status.state = state
        except (AttributeError, TypeError):
            pass

    def _stop_robot(self):
        self.robot.stop()

    # ==========================================================
    # ACTION DEBOUNCE
    # ==========================================================

    def _update_action_history(self, people: Iterable[dict]):
        for person in people:
            pid = self._person_id(person)
            if pid is None:
                continue

            action = self._action(person)
            self._action_history[pid].append(action)

    def _stable_action(self, person) -> str:
        """
        Return the action only after debounce_frames observations.

        If there are not enough observations yet, return the current
        action only when it is a safety action; otherwise unknown.
        """
        pid = self._person_id(person)
        action = self._action(person)

        if pid is None:
            return action

        history = self._action_history[pid]

        if action == "falling":
            return action

        required = max(1, self.config.debounce_frames)

        if len(history) < required:
            return "unknown"

        recent = list(history)[-required:]

        if all(item == action for item in recent):
            return action

        return "unknown"

    # ==========================================================
    # SAFETY
    # ==========================================================

    @staticmethod
    def _landmark_map(landmarks) -> Dict[str, Any]:
        if not landmarks:
            return {}

        if isinstance(landmarks, dict):
            return landmarks

        result = {}

        for lm in landmarks:
            if not isinstance(lm, dict):
                continue

            name = lm.get("name")
            if name:
                result[name] = lm

        return result

    @staticmethod
    def _xy(landmark):
        if landmark is None:
            return None

        if isinstance(landmark, dict):
            if "x" in landmark and "y" in landmark:
                try:
                    return float(landmark["x"]), float(landmark["y"])
                except (TypeError, ValueError):
                    return None

        if isinstance(landmark, (tuple, list)) and len(landmark) >= 2:
            try:
                return float(landmark[0]), float(landmark[1])
            except (TypeError, ValueError):
                return None

        return None

    def check_fall(self, person) -> bool:
        """
        Independent MediaPipe pose-based fall detector.

        A normal standing torso is mostly vertical.
        A fallen torso becomes approximately horizontal.

        This intentionally uses the shoulder/hip geometry rather than
        VideoMAE, so the two safety paths remain independent.
        """
        landmarks = self._landmark_map(person.get("landmarks"))

        ls = self._xy(landmarks.get("LEFT_SHOULDER"))
        rs = self._xy(landmarks.get("RIGHT_SHOULDER"))
        lh = self._xy(landmarks.get("LEFT_HIP"))
        rh = self._xy(landmarks.get("RIGHT_HIP"))

        if not all((ls, rs, lh, rh)):
            return False

        shoulder_center = (
            (ls[0] + rs[0]) / 2.0,
            (ls[1] + rs[1]) / 2.0,
        )

        hip_center = (
            (lh[0] + rh[0]) / 2.0,
            (lh[1] + rh[1]) / 2.0,
        )

        dx = abs(hip_center[0] - shoulder_center[0])
        dy = abs(hip_center[1] - shoulder_center[1])

        # Horizontal torso => likely fall.
        return dx > 0 and dy <= dx * 0.60

    def check_proximity_risk(self, person) -> bool:
        """
        Estimate proximity from bounding-box area.

        The project currently only passes frame_width to DecisionEngine,
        so frame_width² is used as the normalized reference area.
        """
        bbox = self._bbox(person)

        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        area = width * height
        reference_area = float(self.frame_width * self.frame_width)

        ratio = area / reference_area

        return ratio >= self.config.proximity_area_ratio

    def check_tracking_lost(self, frame_idx: int) -> bool:
        """
        True when the current robot target has been missing longer than
        the configured grace period.

        NOTE:
        This must be evaluated before the generic empty-scene/idle path.
        """
        if self.target_id is None:
            return False

        if self.target_id not in self._last_seen_frame:
            return False

        missing_frames = frame_idx - self._last_seen_frame[self.target_id]

        return missing_frames > self.config.lost_tracking_grace

    # ==========================================================
    # ROBOT ACTIONS
    # ==========================================================

    def _target_center_error(self, person) -> float:
        bbox = self._bbox(person)

        if bbox is None:
            return 0.0

        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0

        frame_center = self.frame_width / 2.0

        # -1 = far left, +1 = far right
        return (center_x - frame_center) / frame_center

    def _follow(self, person):
        error = self._target_center_error(person)
        angular = self.config.turn_gain * error

        self.robot.move(
            linear=self.config.follow_linear_speed,
            angular=angular,
        )

        self._set_state(RobotState.FOLLOWING)

    def _approach(self, person):
        error = self._target_center_error(person)
        angular = self.config.turn_gain * error

        self.robot.move(
            linear=self.config.approach_linear_speed,
            angular=angular,
        )

        self._set_state(RobotState.APPROACHING)

    # ==========================================================
    # DECISION STEP
    # ==========================================================

    def step(self, frame_idx: int, people):
        """
        Process one frame.

        Returns a dictionary containing:
            decision
            reason
            target
            detail
        """
        if people is None:
            people = []

        people = list(people)

        # ------------------------------------------------------
        # 1. Emergency latch
        # ------------------------------------------------------
        if getattr(self.robot, "in_emergency", False):
            self._stop_robot()

            return {
                "decision": "holding_emergency_stop",
                "reason": "emergency_latched",
                "target": self.target_id,
                "detail": {},
            }

        # ------------------------------------------------------
        # 2. Update last-seen data BEFORE safety/target logic.
        # ------------------------------------------------------
        visible_ids = set()

        for person in people:
            pid = self._person_id(person)

            if pid is not None:
                visible_ids.add(pid)
                self._last_seen_frame[pid] = frame_idx

        # ------------------------------------------------------
        # 3. Update independent action histories.
        # ------------------------------------------------------
        self._update_action_history(people)

        # ------------------------------------------------------
        # 4. SAFETY: pose fall has highest physical-person priority.
        # ------------------------------------------------------
        for person in people:
            if self.check_fall(person):
                pid = self._person_id(person)

                self._stop_robot()

                self.robot.emergency_stop(
                    reason=f"Pose fall detected for person {pid}"
                )

                return {
                    "decision": "emergency_stop",
                    "reason": "pose_fall",
                    "target": pid,
                    "detail": {
                        "id": pid,
                    },
                }

        # ------------------------------------------------------
        # 5. SAFETY: VideoMAE falling action.
        # ------------------------------------------------------
        for person in people:
            action = self._action(person)
            confidence = self._confidence(person)

            if (
                action == "falling"
                and confidence >= self.config.fall_action_confidence
            ):
                pid = self._person_id(person)

                self._stop_robot()

                self.robot.emergency_stop(
                    reason=(
                        f"VideoMAE detected falling person "
                        f"{pid} with confidence {confidence:.2f}"
                    )
                )

                # IMPORTANT:
                # Tests and the rest of the project use "action_fall".
                return {
                    "decision": "emergency_stop",
                    "reason": "action_fall",
                    "target": pid,
                    "detail": {
                        "id": pid,
                        "confidence": confidence,
                    },
                }

        # ------------------------------------------------------
        # 6. SAFETY: proximity risk.
        # ------------------------------------------------------
        for person in people:
            if self.check_proximity_risk(person):
                pid = self._person_id(person)

                self._stop_robot()

                self.robot.emergency_stop(
                    reason=f"person {pid} too close to robot"
                )

                # IMPORTANT:
                # Keep the public reason name expected by tests.
                return {
                    "decision": "emergency_stop",
                    "reason": "proximity_risk",
                    "target": pid,
                    "detail": {
                        "id": pid,
                    },
                }

        # ------------------------------------------------------
        # 7. TRACKING LOSS
        #
        # THIS IS THE MAIN BUG FIX.
        #
        # Do this BEFORE:
        #
        #     if not people:
        #         return idle
        #
        # Otherwise the target disappears and the engine reports
        # "idle" forever instead of "stop_target_lost".
        # ------------------------------------------------------
        if self.check_tracking_lost(frame_idx):
            lost_id = self.target_id

            self._stop_robot()
            self._set_state(RobotState.STOPPED)

            return {
                "decision": "stop_target_lost",
                "reason": "tracking_lost",
                "target": lost_id,
                "detail": {
                    "id": lost_id,
                    "last_seen_frame": self._last_seen_frame.get(lost_id),
                    "current_frame": frame_idx,
                    "grace_frames": self.config.lost_tracking_grace,
                },
            }

        # ------------------------------------------------------
        # 8. No people.
        #
        # Safe because tracking-loss was already checked above.
        # ------------------------------------------------------
        if not people:
            self._stop_robot()
            self._set_state(RobotState.IDLE)

            return {
                "decision": "idle",
                "reason": "no_people",
                "target": self.target_id,
                "detail": {},
            }

        # ------------------------------------------------------
        # 9. Assistance has priority over following.
        # ------------------------------------------------------
        assist_actions = {
            "waving",
            "help_request",
            "distress_gesture",
        }

        for person in people:
            stable = self._stable_action(person)

            if stable in assist_actions:
                pid = self._person_id(person)

                self.target_id = pid
                self.target_action = stable

                self._approach(person)

                return {
                    "decision": "approach_assist",
                    "reason": stable,
                    "target": pid,
                    "detail": {
                        "id": pid,
                        "action": stable,
                    },
                }

        # ------------------------------------------------------
        # 10. Following.
        # ------------------------------------------------------
        for person in people:
            stable = self._stable_action(person)

            if stable == "walking":
                pid = self._person_id(person)

                self.target_id = pid
                self.target_action = stable

                self._follow(person)

                return {
                    "decision": "follow",
                    "reason": "walking",
                    "target": pid,
                    "detail": {
                        "id": pid,
                        "action": stable,
                    },
                }

        # ------------------------------------------------------
        # 11. Unknown/idle action.
        # ------------------------------------------------------
        self._stop_robot()
        self._set_state(RobotState.IDLE)

        return {
            "decision": "idle",
            "reason": "no_action",
            "target": self.target_id,
            "detail": {},
        }