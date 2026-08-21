"""
web_pipeline.py

Member 3 + Member 4 integration - WEB VIEW

Same pipeline as robot_pipeline.py:

    YOLO -> Tracker -> MediaPipe -> ActionBuffer -> VideoMAE
         -> ActionSmoother -> DecisionEngine -> Robot

but instead of a cv2.imshow() window, the annotated frame and the
robot/decision/action status are served over HTTP so you can watch
everything from a browser (useful for headless machines, remote
viewing, or just not fighting with an OpenCV window).

Run:
    python web_pipeline.py --source 0 --device cuda

Then open:
    http://127.0.0.1:5000

Requires:
    pip install flask
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from detection import PersonDetector
from pose_extraction import PoseExtractor
from tracker import CentroidTracker
from visualization import draw_tracked_people

from action_buffer import ActionBuffer
from action_recognizer import VideoMAEActionRecognizer
from action_smoother import ActionSmoother
from safety import infer_fallback_action

from robot_interface import SimulatedRobot, SerialRobot, RobotState
from decision_engine import DecisionEngine, DecisionConfig


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Robot action pipeline with a live web dashboard"
    )

    p.add_argument("--source", default="0")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="cuda")

    p.add_argument("--robot", choices=["sim", "serial"], default="sim")
    p.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for --robot serial")
    p.add_argument("--log", default="robot_run.log")

    p.add_argument("--host", default="127.0.0.1", help="Web dashboard host")
    p.add_argument("--web-port", type=int, default=5000, help="Web dashboard port")

    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for the video stream (1-100)",
    )

    return p.parse_args()


# ==========================================================
# SHARED STATE (written by capture thread, read by Flask)
# ==========================================================

class SharedState:

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes = None
        self.status = {
            "robot_state": "unknown",
            "decision": "starting",
            "frame_idx": 0,
            "persons": [],
            "in_emergency": False,
        }
        self.stop_event = threading.Event()


state = SharedState()
robot_ref = {"robot": None}  # populated once robot is created


# ==========================================================
# ROBOT STATE COLORS (BGR, for the video overlay)
# ==========================================================

STATE_COLORS = {
    RobotState.IDLE: (200, 200, 200),
    RobotState.FOLLOWING: (0, 255, 0),
    RobotState.APPROACHING: (0, 200, 255),
    RobotState.ASSISTING: (0, 165, 255),
    RobotState.STOPPED: (0, 0, 255),
    RobotState.EMERGENCY_STOP: (0, 0, 255),
}


def overlay_robot_status(frame, robot, decision):

    text = f"ROBOT: {robot.status.state.value} | {decision.get('decision', '')}"

    # Emergency behavior is unchanged either way -- this is just a
    # visual nudge for the operator deciding whether to clear it.
    if decision.get("likely_sleeping"):
        text += " (possibly asleep -- operator review)"

    color = STATE_COLORS.get(robot.status.state, (255, 255, 255))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )
    return frame


def draw_action_label(frame, person):

    x1, y1, x2, y2 = person["bbox"]
    person_id = person["id"]
    action = person.get("action", "unknown")
    confidence = float(person.get("action_confidence", 0.0) or 0.0)

    # "unknown" means VideoMAE's raw Kinetics label just isn't one the
    # robot acts on -- show that raw label instead of the dead-end
    # "unknown" so it's still useful for eyeballing what the model saw.
    raw_action = person.get("raw_action")
    if action == "unknown" and raw_action and raw_action != "unknown":
        raw_confidence = float(person.get("raw_confidence", 0.0) or 0.0)
        text = f"ID {person_id}: unknown ({raw_action} {raw_confidence:.2f})"
    elif person.get("action_source") == "pose":
        text = f"ID {person_id}: {action} {confidence:.2f} (pose)"
    else:
        text = f"ID {person_id}: {action} {confidence:.2f}"

    text_x = int(x1)
    text_y = max(20, int(y1) - 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    bg_x1 = text_x
    bg_y1 = max(0, text_y - text_height - 6)
    bg_x2 = text_x + text_width + 8
    bg_y2 = text_y + baseline + 3

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (text_x + 4, text_y),
        font, font_scale, (0, 255, 255), thickness,
    )


# ==========================================================
# CAPTURE / PIPELINE LOOP (runs in a background thread)
# ==========================================================

def capture_loop(args):

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

    # ------------------------------------------------------
    # MEMBER 2
    # ------------------------------------------------------

    detector = PersonDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )
    pose_extractor = PoseExtractor()
    tracker = CentroidTracker()

    # ------------------------------------------------------
    # MEMBER 3
    # ------------------------------------------------------

    SEQUENCE_LENGTH = 16
    INFERENCE_INTERVAL = 8

    # Confidence assigned to actions filled in by the pose-geometry
    # fallback (see PROCESS EACH PERSON below) when VideoMAE has
    # nothing usable. Deliberately below action_fall_confidence so it
    # can never itself satisfy a VideoMAE-fall emergency check.
    POSE_FALLBACK_CONFIDENCE = 0.5

    action_buffer = ActionBuffer(sequence_length=SEQUENCE_LENGTH)
    action_recognizer = VideoMAEActionRecognizer(
        sequence_length=SEQUENCE_LENGTH,
        device=args.device,
    )

    smoothers = {}
    last_actions = {}

    # ------------------------------------------------------
    # MEMBER 4
    # ------------------------------------------------------

    if args.robot == "sim":
        robot = SimulatedRobot(log_path=args.log)
    else:
        robot = SerialRobot(port=args.port)

    robot_ref["robot"] = robot

    engine = DecisionEngine(
        robot,
        frame_width=width,
        config=DecisionConfig(
            proximity_area_ratio=0.9,
            action_fall_confidence=0.60,
        ),
    )

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_idx = 0

    print()
    print("======================================")
    print("WEB ROBOT ACTION PIPELINE")
    print("======================================")
    print(f"Dashboard: http://{args.host}:{args.web_port}")
    print("======================================")
    print()

    try:

        while not state.stop_event.is_set():

            ok, frame = cap.read()
            if not ok:
                print("Video source ended.")
                break

            frame_idx += 1

            # ==================================================
            # 1. YOLO
            # ==================================================

            detections = detector.detect(frame)

            # ==================================================
            # 2. TRACKER
            # ==================================================

            tracked_people = tracker.update(detections)
            current_ids = set()

            # ==================================================
            # 3. PROCESS EACH PERSON
            # ==================================================

            for person in tracked_people:

                person_id = person["id"]
                current_ids.add(person_id)

                crop, offset = detector.crop_person(frame, person["bbox"])
                person["landmarks"] = pose_extractor.extract(crop, offset)

                if person_id not in smoothers:
                    smoothers[person_id] = ActionSmoother(
                        window_size=7,
                        min_frame_confidence=0.03,
                        stable_confidence_threshold=0.15,
                    )

                if person_id not in last_actions:
                    last_actions[person_id] = {
                        "action": "unknown",
                        "confidence": 0.0,
                        "raw_action": "unknown",
                        "raw_confidence": 0.0,
                    }

                sequence = action_buffer.add_frame(person_id, frame)

                # Persist last known action between VideoMAE cycles.
                person["action"] = last_actions[person_id]["action"]
                person["action_confidence"] = last_actions[person_id]["confidence"]
                person["raw_action"] = last_actions[person_id]["raw_action"]
                person["raw_confidence"] = last_actions[person_id]["raw_confidence"]

                # ==================================================
                # 3b. POSE-BASED ACTION FALLBACK
                #
                # VideoMAE's Kinetics-400 checkpoint frequently lands
                # on labels outside ACTION_MAP (e.g. "stretching arm"),
                # leaving the action stuck on "unknown" between
                # inference cycles -- in practice this meant follow /
                # approach almost never fired. Cheap per-frame pose
                # geometry fills that gap for walking/waving every
                # frame (no model call needed). It never overrides a
                # real VideoMAE action, and deliberately excludes
                # "falling" -- that stays the job of the dedicated,
                # independent check_fall() safety path so there's
                # exactly one source of truth for the emergency-stop
                # trigger.
                # ==================================================

                person["action_source"] = "videomae"

                if person["action"] == "unknown":
                    pose_action = infer_fallback_action(person["landmarks"])
                    if pose_action in ("walking", "waving", "fixing_hair"):
                        person["action"] = pose_action
                        person["action_confidence"] = POSE_FALLBACK_CONFIDENCE
                        person["action_source"] = "pose"

                # ==================================================
                # 4. VIDEOMAE
                # ==================================================

                if sequence is not None and frame_idx % INFERENCE_INTERVAL == 0:

                    try:
                        result = action_recognizer.predict(sequence)

                        raw_action = result["action"]
                        raw_confidence = float(result["confidence"])

                        stable_result = smoothers[person_id].update(
                            raw_action, raw_confidence
                        )

                        stable_action = stable_result["action"]
                        stable_confidence = float(stable_result["confidence"])

                        raw_model_action = result.get("raw_action", raw_action)

                        last_actions[person_id] = {
                            "action": stable_action,
                            "confidence": stable_confidence,
                            "raw_action": raw_model_action,
                            "raw_confidence": raw_confidence,
                        }

                        person["action"] = stable_action
                        person["action_confidence"] = stable_confidence
                        person["raw_action"] = raw_model_action
                        person["raw_confidence"] = raw_confidence
                        person["action_source"] = "videomae"

                    except Exception as e:
                        print(f"[Person {person_id}] VideoMAE error: {e}")

            # ==================================================
            # 5. CLEANUP LOST PEOPLE
            # ==================================================

            known_ids = set(action_buffer.buffers.keys())
            for person_id in known_ids:
                if person_id not in current_ids:
                    action_buffer.remove_person(person_id)
                    smoothers.pop(person_id, None)
                    last_actions.pop(person_id, None)

            # ==================================================
            # 6. DECISION ENGINE
            # ==================================================

            decision = engine.step(frame_idx, tracked_people)

            # ==================================================
            # 7. VISUALIZATION
            # ==================================================

            annotated = draw_tracked_people(frame.copy(), tracked_people)
            annotated = overlay_robot_status(annotated, robot, decision)

            for person in tracked_people:
                draw_action_label(annotated, person)

            # ==================================================
            # 8. PUBLISH TO WEB DASHBOARD
            # ==================================================

            ok_encode, buffer = cv2.imencode(".jpg", annotated, encode_params)

            persons_status = [
                {
                    "id": p["id"],
                    "action": p.get("action", "unknown"),
                    "confidence": round(float(p.get("action_confidence", 0.0) or 0.0), 3),
                    "raw_action": p.get("raw_action", "unknown"),
                    "raw_confidence": round(float(p.get("raw_confidence", 0.0) or 0.0), 3),
                    "action_source": p.get("action_source", "videomae"),
                    "bbox": list(p["bbox"]),
                }
                for p in tracked_people
            ]

            with state.lock:
                if ok_encode:
                    state.jpeg_bytes = buffer.tobytes()
                state.status = {
                    "robot_state": robot.status.state.value,
                    "decision": decision.get("decision", ""),
                    "reason": decision.get("reason", ""),
                    "frame_idx": frame_idx,
                    "persons": persons_status,
                    "in_emergency": bool(robot.in_emergency),
                    "likely_sleeping": bool(decision.get("likely_sleeping", False)),
                }

    finally:
        cap.release()
        pose_extractor.close()
        if args.robot == "sim":
            robot.close()

        print()
        print(f"Processed {frame_idx} frames.")
        print(f"Final robot state: {robot.status.state.value}")


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Robot Action Pipeline - Live Dashboard</title>
  <style>
    body {
      background: #111;
      color: #eee;
      font-family: -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 0;
      padding: 24px;
    }
    h1 { font-size: 20px; margin-bottom: 16px; color: #fafafa; }
    .layout {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      align-items: flex-start;
    }
    .video-panel img {
      max-width: 720px;
      width: 100%;
      border-radius: 8px;
      border: 1px solid #333;
      display: block;
    }
    .status-panel {
      background: #1b1b1b;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 16px;
      min-width: 280px;
      flex: 1;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 13px;
      margin-bottom: 12px;
    }
    .badge.idle { background: #444; color: #ddd; }
    .badge.following { background: #1e5c2f; color: #b6ffcb; }
    .badge.approaching { background: #6a4e00; color: #ffe3a3; }
    .badge.assisting { background: #6a3f00; color: #ffd8a3; }
    .badge.stopped, .badge.emergency_stop { background: #6a1414; color: #ffb3b3; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { text-align: left; padding: 6px 4px; font-size: 13px; border-bottom: 1px solid #2a2a2a; }
    th { color: #888; font-weight: 500; }
    button {
      background: #6a1414;
      color: #ffdddd;
      border: none;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      margin-top: 10px;
    }
    button:hover { background: #841919; }
    .meta { color: #888; font-size: 12px; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>Robot Action Pipeline &mdash; Live Dashboard</h1>
  <div class="layout">
    <div class="video-panel">
      <img src="/video_feed" alt="live feed">
    </div>
    <div class="status-panel">
      <div id="badge" class="badge idle">IDLE</div>
      <div><strong>Decision:</strong> <span id="decision">-</span></div>
      <div><strong>Reason:</strong> <span id="reason">-</span></div>
      <table>
        <thead>
          <tr><th>ID</th><th>Action</th><th>Conf.</th><th>Source</th><th>Raw label</th></tr>
        </thead>
        <tbody id="persons"></tbody>
      </table>
      <button id="clear-btn" onclick="clearEmergency()">Clear Emergency</button>
      <div class="meta" id="frame-meta">frame 0</div>
    </div>
  </div>

  <script>
    async function refresh() {
      try {
        const res = await fetch('/status');
        const data = await res.json();

        const badge = document.getElementById('badge');
        badge.textContent = data.robot_state.toUpperCase();
        badge.className = 'badge ' + data.robot_state.toLowerCase();

        document.getElementById('decision').textContent = data.decision || '-';
        document.getElementById('reason').textContent = data.reason || '-';
        document.getElementById('frame-meta').textContent =
          'frame ' + data.frame_idx
          + (data.in_emergency ? ' | EMERGENCY LATCHED' : '')
          + (data.likely_sleeping ? ' | possibly asleep -- operator review' : '');

        const tbody = document.getElementById('persons');
        tbody.innerHTML = '';
        data.persons.forEach(p => {
          const tr = document.createElement('tr');
          const rawCell = (p.action === 'unknown' && p.raw_action && p.raw_action !== 'unknown')
            ? `${p.raw_action} (${p.raw_confidence.toFixed(2)})`
            : '-';
          [p.id, p.action, p.confidence.toFixed(2), p.action_source || 'videomae', rawCell].forEach(value => {
            const td = document.createElement('td');
            td.textContent = value;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error('status refresh failed', e);
      }
    }

    async function clearEmergency() {
      await fetch('/emergency/clear', { method: 'POST' });
      refresh();
    }

    setInterval(refresh, 300);
    refresh();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


def mjpeg_generator():
    while not state.stop_event.is_set():
        with state.lock:
            frame_bytes = state.jpeg_bytes
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(1 / 30)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with state.lock:
        return jsonify(state.status)


@app.route("/emergency/clear", methods=["POST"])
def clear_emergency():
    robot = robot_ref.get("robot")
    if robot is not None:
        robot.clear_emergency()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "robot not ready"}), 503


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    args = parse_args()

    worker = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    worker.start()

    try:
        app.run(host=args.host, port=args.web_port, threaded=True)
    finally:
        state.stop_event.set()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()