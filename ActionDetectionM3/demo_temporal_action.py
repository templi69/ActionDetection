import cv2

from action_buffer import ActionBuffer
from action_recognizer import VideoMAEActionRecognizer


VIDEO_PATH = "test2.mp4"

buffer = ActionBuffer(sequence_length=16)

recognizer = VideoMAEActionRecognizer(
    sequence_length=16
)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    sequence = buffer.add_frame(
        person_id=0,
        frame=frame
    )

    if sequence is not None:

        result = recognizer.predict(sequence)

        print("\n========== ACTION RESULT ==========")
        print(f"Action:     {result['action']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Class ID:   {result['class_id']}")

cap.release()