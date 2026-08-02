# Member 4 — Decision Logic & Robot Integration

## Files

| File | Purpose |
|---|---|
| `robot_interface.py` | Abstract robot API (`move`, `stop`, `emergency_stop`, `alert`) plus a `SimulatedRobot` (logs + fake kinematics — no hardware needed) and a `SerialRobot` stub for when real hardware is ready. |
| `safety.py` | Pure functions: fall detection (torso angle from pose landmarks), proximity/collision risk (bbox size vs. frame), lost-tracking detection. Independent of Member 3's action classifier — a safety net even if that model is wrong or missing. |
| `decision_engine.py` | The rule engine. Every frame: emergency stop → assistance request → follow → idle, with debouncing so one noisy frame doesn't flip the robot's behavior. |
| `robot_pipeline.py` | End-to-end runner: plugs into Member 2's `detection.py` / `pose_extraction.py` / `tracker.py` / `visualization.py` and drives a robot live from webcam or video. |
| `test_decision_engine.py` | Pytest suite covering the decision layer with synthetic data — no camera, YOLO, or MediaPipe required. |

## How it fits with the rest of the team

- **Member 2** gives you tracked people per frame: `{"id", "bbox", "landmarks"}`.
- **Member 3** (action recognition) is expected to add `"action"` (e.g. `"walking"`, `"waving"`, `"falling"`) and `"action_confidence"` to each of those dicts.
- **You (Member 4)** consume that combined structure and decide what the robot does.

If Member 3's classifier isn't ready yet, `safety.infer_fallback_action()` derives a rough
action straight from the landmarks so you can build and test this whole layer independently
today. When Member 3's real model lands, just replace that one line in `robot_pipeline.py`:

```python
person["action"] = infer_fallback_action(person["landmarks"])
# becomes:
person["action"] = action_recognizer.predict(person_sequence)
```

Nothing else in `decision_engine.py` needs to change — it only cares about the `action` string.

## Decision rules (priority order, evaluated every frame)

1. **Emergency stop** — a fall is detected (torso geometry near-horizontal) OR a person's
   bounding box is too large relative to the frame (collision risk). This latches: the robot
   ignores further `move()` calls until an operator calls `clear_emergency()`.
2. **Assistance** — a stable `"waving"` / `"distress_gesture"` action → robot turns to face
   and approaches the person, and raises an alert.
3. **Follow** — a stable `"walking"` action on the current or a new target → robot follows,
   steering based on how far the person's bbox center is from the frame's horizontal center.
4. **Idle** — nobody actionable → robot holds position.

"Stable" means the action won majority vote over the last `debounce_frames` frames
(default 5), so a single misclassified frame can't yank the robot around.

## Running it

```bash
# Install deps (shared with Member 2's environment)
pip install ultralytics mediapipe opencv-python pytest

# Live webcam demo, simulated robot, with preview window
python robot_pipeline.py --source 0

# Batch process a video, no preview (e.g. on a headless machine)
python robot_pipeline.py --source video.mp4 --no-display

# Against real hardware once wired up
python robot_pipeline.py --source 0 --robot serial --port /dev/ttyUSB0
```

While the preview window is open: `q` quits, `c` manually clears an emergency stop
(simulates an operator override, useful for testing the fall/proximity triggers live).

Every run also writes `robot_run.log` (simulated robot) with a timestamped trace of every
move/stop/alert command — useful for your report and for replaying a demo run.

## Testing end-to-end

Two levels, since you don't want to depend on a webcam + GPU just to check your logic:

1. **Logic-only (fast, no camera/model dependencies):**
   ```bash
   pytest test_decision_engine.py -v
   ```
   This feeds synthetic pose data straight into `DecisionEngine` + `SimulatedRobot` and checks:
   idle when empty, follow-after-debounce, approach-on-wave, emergency-stop-on-fall-geometry,
   emergency latching/clearing, proximity-triggered stop, and lost-target stop.

2. **Full pipeline (needs Member 2's modules + a webcam or sample video):**
   ```bash
   python robot_pipeline.py --source sample_video.mp4
   ```
   Watch the overlay in the top-left corner (robot state + decision each frame) and confirm
   it matches what's happening in the video — e.g. someone walking → "following", someone
   waving → "approaching", someone lying down → "emergency_stop".

For the group demo, a good end-to-end test clip includes: someone walking toward/past the
camera, someone waving, and someone lying down (staged fall) — that exercises all four
branches of the decision engine in one recording.
