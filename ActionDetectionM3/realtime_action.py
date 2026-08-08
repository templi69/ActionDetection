
import cv2
import time

from action_smoother import ActionSmoother
from action_buffer import ActionBuffer
from action_recognizer import VideoMAEActionRecognizer


# 0 = default webcam
CAMERA_ID = 0

SEQUENCE_LENGTH = 16

# Run VideoMAE every N frames.
# Higher value = less GPU usage.
INFERENCE_INTERVAL = 8


def main():

    print("Starting real-time action recognition...")

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Buffer for one camera stream
    buffer = ActionBuffer(
        sequence_length=SEQUENCE_LENGTH
    )

    # Action smoother
    smoother = ActionSmoother(
        window_size=7,
        confidence_threshold=0.15
    )

    # VideoMAE
    recognizer = VideoMAEActionRecognizer(
        sequence_length=SEQUENCE_LENGTH
    )

    frame_count = 0

    current_action = "Waiting..."
    current_confidence = 0.0

    fps_start = time.time()
    fps_counter = 0
    fps = 0.0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        frame_count += 1
        fps_counter += 1

        # Add frame to temporal buffer
        sequence = buffer.add_frame(
            person_id=0,
            frame=frame
        )

        # Run VideoMAE periodically
        if (
            sequence is not None
            and frame_count % INFERENCE_INTERVAL == 0
        ):

            try:

                # Raw VideoMAE prediction
                result = recognizer.predict(
                    sequence
                )

                raw_action = result["action"]
                raw_confidence = result["confidence"]

                # Send prediction through smoother
                stable_result = smoother.update(
                    raw_action,
                    raw_confidence
                )

                current_action = stable_result["action"]
                current_confidence = stable_result["confidence"]

                print(
                    f"Raw: {raw_action} "
                    f"({raw_confidence:.3f}) | "
                    f"Stable: {current_action} "
                    f"({current_confidence:.3f})"
                )

            except Exception as e:

                print(
                    f"VideoMAE error: {e}"
                )

        # Calculate FPS
        elapsed = time.time() - fps_start

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_counter = 0
            fps_start = time.time()

        # Draw stable action
        cv2.putText(
            frame,
            f"Action: {current_action}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show camera
        cv2.imshow(
            "Real-Time VideoMAE Action Recognition",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Real-time action recognition stopped.")


if __name__ == "__main__":
    main()
