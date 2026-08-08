import numpy as np

from action_buffer import ActionBuffer


buffer = ActionBuffer(sequence_length=16)

for frame_number in range(20):

    # Fake 640x480 OpenCV frame
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    sequence = buffer.add_frame(
        person_id=0,
        frame=frame
    )

    if sequence is not None:
        print(
            f"Frame {frame_number + 1}: "
            f"Sequence ready → {len(sequence)} frames"
        )