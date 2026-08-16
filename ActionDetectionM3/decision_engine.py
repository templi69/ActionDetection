"""
decision_engine.py

Member 4 - Decision Logic

Maps recognized person actions + safety signals
to robot commands.

Pipeline input per person:

{
    "id": int,
    "bbox": (x1, y1, x2, y2),
    "landmarks": [...] or None,
    "action": str,
    "action_confidence": float,
}

Priority:

1. Emergency / fall
2. Collision / proximity
3. Emergency latch
4. Assistance request
5. Following
6. Idle
"""

from collections import deque, Counter
from dataclasses import dataclass

from robot_interface import RobotState

from safety import (
    check_fall,
    check_proximity_risk,
    check_tracking_lost,
    is_horizontal,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

@dataclass
class DecisionConfig:

    # Number of consecutive action labels required
    # before the action is accepted.
    debounce_frames: int = 5

    # Bounding-box area ratio considered too close.
    proximity_area_ratio: float = 0.35

    # Number of frames a target can disappear.
    lost_tracking_grace: int = 20

    # Robot movement speeds.
    follow_linear_speed: float = 0.4
    approach_linear_speed: float = 0.25

    # Turning sensitivity.
    turn_gain: float = 0.003

    # ------------------------------------------------------
    # VideoMAE falling confidence threshold.
    #
    # Example:
    #
    # falling + confidence >= 0.60
    #
    # -> emergency stop
    # ------------------------------------------------------

    action_fall_confidence: float = 0.60

    # ------------------------------------------------------
    # Sleeping vs. fallen.
    #
    # Kinetics-400 has no "sleeping" class, and the only real signal
    # for "lying down" is the same torso-horizontal geometry used to
    # trigger the pose fall emergency. So instead of changing when
    # emergency_stop fires, DecisionEngine tracks how long a person
    # has stayed horizontal AND roughly still; past this many frames
    # the fall decision is additionally flagged "likely_sleeping" for
    # an operator to see. This NEVER auto-clears the emergency latch
    # -- a human still has to call robot.clear_emergency().
    #
    # ~8s at 30fps / one DecisionEngine.step() call per frame.
    # ------------------------------------------------------

    sleep_still_frames: int = 240

    # Max per-frame bbox-centroid movement (px) to still count as
    # "still". Bigger movement resets the counter -- thrashing/
    # distress shouldn't read as calmly sleeping.
    sleep_motion_threshold: float = 15.0


# ==========================================================
# DECISION ENGINE
# ==========================================================

class DecisionEngine:

    """
    Robot decision-making layer.

    Priority order:

    1. Pose-based fall
    2. VideoMAE falling with sufficient confidence
    3. Proximity / collision risk
    4. Existing emergency latch
    5. Assistance request
    6. Walking / following
    7. Idle
    """

    def __init__(
        self,
        robot,
        frame_width: int,
        config: DecisionConfig = None
    ):

        self.robot = robot

        self.frame_width = frame_width

        self.cfg = (
            config
            or DecisionConfig()
        )

        # --------------------------------------------------
        # Person ID -> recent actions
        # --------------------------------------------------

        self._action_history = {}

        # --------------------------------------------------
        # Person ID -> last frame seen
        # --------------------------------------------------

        self._last_seen_frame = {}

        # --------------------------------------------------
        # Current primary target
        # --------------------------------------------------

        self._primary_target_id = None

        # --------------------------------------------------
        # Person ID -> {"frames": int, "last_centroid": (x, y)}
        # Consecutive horizontal-and-still frames, for the
        # sleeping-vs-fallen distinction. See DecisionConfig.
        # --------------------------------------------------

        self._horizontal_tracking = {}


    # ======================================================
    # SLEEP TRACKING
    #
    # Updates (and returns) how many consecutive frames this person
    # has been horizontal AND roughly still. Called for every tracked
    # person every frame, independent of whether check_fall currently
    # flags them -- so the count is accurate the instant it's needed.
    # ======================================================

    def _update_sleep_tracking(self, person):

        pid = person["id"]

        horizontal = is_horizontal(person.get("landmarks"))

        if not horizontal:
            self._horizontal_tracking.pop(pid, None)
            return 0

        x1, y1, x2, y2 = person["bbox"]
        centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        entry = self._horizontal_tracking.get(pid)

        if entry is None:
            self._horizontal_tracking[pid] = {"frames": 1, "last_centroid": centroid}
            return 1

        last_centroid = entry["last_centroid"]

        moved = (
            (centroid[0] - last_centroid[0]) ** 2
            + (centroid[1] - last_centroid[1]) ** 2
        ) ** 0.5

        if moved > self.cfg.sleep_motion_threshold:
            entry["frames"] = 1
        else:
            entry["frames"] += 1

        entry["last_centroid"] = centroid

        return entry["frames"]


    # ======================================================
    # MAIN DECISION FUNCTION
    # ======================================================

    def step(
        self,
        frame_idx,
        tracked_people
    ):

        """
        Process all tracked people and issue
        the appropriate robot command.
        """

        # ==================================================
        # 1. POSE-BASED FALL
        #
        # This is safety-critical and does NOT depend
        # on VideoMAE. Emergency behavior below is unchanged
        # regardless of how long someone has been down --
        # "likely_sleeping" is informational only, for an
        # operator deciding whether to clear the emergency.
        # ==================================================

        for person in tracked_people:
            self._update_sleep_tracking(person)

        fallen = check_fall(
            tracked_people
        )

        if fallen:

            still_frames = (
                self._horizontal_tracking
                .get(fallen["id"], {})
                .get("frames", 0)
            )

            likely_sleeping = (
                still_frames >= self.cfg.sleep_still_frames
            )

            reason = (
                f"person {fallen['id']} "
                f"appears to have fallen "
                f"(torso angle "
                f"{fallen['torso_angle_deg']:.1f}°)"
            )

            if likely_sleeping:
                reason += " -- lying still, possibly asleep"

            self.robot.emergency_stop(
                reason=reason
            )

            return {

                "decision":
                    "emergency_stop",

                "reason":
                    "pose_fall",

                "detail":
                    fallen,

                "likely_sleeping":
                    likely_sleeping,

                "horizontal_still_frames":
                    still_frames

            }


        # ==================================================
        # 2. VIDEOMAE FALL
        #
        # Only trigger when confidence >= 0.60.
        # ==================================================

        for person in tracked_people:

            action = (
                person.get(
                    "action",
                    "unknown"
                )
                or "unknown"
            )

            confidence = float(
                person.get(
                    "action_confidence",
                    0.0
                )
                or 0.0
            )

            if (

                action == "falling"

                and

                confidence >=
                self.cfg.action_fall_confidence

            ):

                self.robot.emergency_stop(

                    reason=(

                        f"VideoMAE detected "
                        f"falling person "
                        f"{person['id']} "
                        f"with confidence "
                        f"{confidence:.2f}"

                    )

                )

                return {

                    "decision":
                        "emergency_stop",

                    "reason":
                        "videomae_fall",

                    "target":
                        person["id"],

                    "confidence":
                        confidence

                }


        # ==================================================
        # 3. PROXIMITY / COLLISION RISK
        # ==================================================

        collision_risk = (
            check_proximity_risk(

                tracked_people,

                self.frame_width,

                self.cfg.proximity_area_ratio

            )
        )

        if collision_risk:

            self.robot.emergency_stop(

                reason=(

                    f"person "
                    f"{collision_risk['id']} "
                    f"too close to robot"

                )

            )

            return {

                "decision":
                    "emergency_stop",

                "reason":
                    "proximity",

                "detail":
                    collision_risk

            }


        # ==================================================
        # 4. EMERGENCY LATCH
        # ==================================================

        if self.robot.in_emergency:

            return {

                "decision":
                    "holding_emergency_stop"

            }


        # ==================================================
        # 5. UPDATE TRACKING / ACTION HISTORY
        # ==================================================

        seen_ids = set()


        for person in tracked_people:

            pid = person["id"]

            seen_ids.add(pid)

            self._last_seen_frame[
                pid
            ] = frame_idx


            history = (
                self._action_history.setdefault(

                    pid,

                    deque(
                        maxlen=
                        self.cfg.debounce_frames
                    )

                )
            )


            history.append(

                person.get(
                    "action",
                    "unknown"
                )

            )


        # ==================================================
        # 6. CHECK PRIMARY TARGET LOST
        # ==================================================

        if (

            self._primary_target_id
            is not None

            and

            self._primary_target_id
            not in seen_ids

        ):

            lost = check_tracking_lost(

                self._primary_target_id,

                frame_idx,

                self._last_seen_frame,

                self.cfg.lost_tracking_grace

            )


            if lost:

                self.robot.stop(
                    reason="tracked person lost"
                )

                lost_id = (
                    self._primary_target_id
                )

                self._primary_target_id = None


                return {

                    "decision":
                        "stop_target_lost",

                    "target":
                        lost_id

                }


        # ==================================================
        # 7. ASSISTANCE
        # ==================================================

        for person in tracked_people:

            stable_action = (
                self._stable_action(
                    person["id"]
                )
            )


            if stable_action in (

                "waving",

                "distress_gesture",

                "help_request"

            ):

                self._primary_target_id = (
                    person["id"]
                )


                self._drive_towards(

                    person,

                    self.cfg.approach_linear_speed

                )


                self.robot.set_state(

                    RobotState.APPROACHING

                )


                self.robot.alert(

                    f"Assisting person "
                    f"{person['id']}"

                )


                return {

                    "decision":
                        "approach_assist",

                    "target":
                        person["id"],

                    "action":
                        stable_action

                }


        # ==================================================
        # 8. FOLLOWING
        # ==================================================

        for person in tracked_people:

            stable_action = (
                self._stable_action(
                    person["id"]
                )
            )


            if (

                stable_action == "walking"

                and

                (
                    self._primary_target_id
                    is None

                    or

                    self._primary_target_id
                    == person["id"]
                )

            ):

                self._primary_target_id = (
                    person["id"]
                )


                self._drive_towards(

                    person,

                    self.cfg.follow_linear_speed

                )


                self.robot.set_state(

                    RobotState.FOLLOWING

                )


                return {

                    "decision":
                        "follow",

                    "target":
                        person["id"]

                }


        # ==================================================
        # 9. IDLE
        # ==================================================

        self.robot.stop(
            reason="no actionable person"
        )


        self.robot.set_state(
            RobotState.IDLE
        )


        return {

            "decision":
                "idle"

        }


    # ======================================================
    # STABLE ACTION
    # ======================================================

    def _stable_action(
        self,
        person_id
    ):

        history = (
            self._action_history.get(
                person_id
            )
        )


        if (

            not history

            or

            len(history)
            < self.cfg.debounce_frames

        ):

            return "unknown"


        return Counter(
            history
        ).most_common(1)[0][0]


    # ======================================================
    # DRIVE TOWARDS PERSON
    # ======================================================

    def _drive_towards(
        self,
        person,
        linear_speed
    ):

        x1, y1, x2, y2 = (
            person["bbox"]
        )


        # Person center.

        cx = (
            x1 + x2
        ) / 2.0


        # Distance from image center.

        offset = (

            cx

            -

            (
                self.frame_width
                / 2.0
            )

        )


        # Turn toward person.

        angular = (

            -self.cfg.turn_gain
            * offset

        )


        self.robot.move(

            linear=linear_speed,

            angular=angular,

            reason=(
                f"tracking id "
                f"{person['id']}"
            )

        )