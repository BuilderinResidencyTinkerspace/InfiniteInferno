import time

from camera import Camera
from preprocess import preprocess
from detector import Detector
from postprocess import postprocess

cam = Camera()

detector = Detector()

frame_count = 0
detections = []

prev_time = time.time()

print("Starting detector...")

try:

    while True:

        frame = cam.read()

        frame_count += 1

        if frame_count % 3 == 0:

            tensor = preprocess(frame)

            output = detector.infer(tensor)

            detections = postprocess(
                output,
                frame.shape[1],
                frame.shape[0]
            )

        current_time = time.time()

        fps = 1 / (current_time - prev_time)

        prev_time = current_time

        print(
            f"\rFPS: {fps:.1f} | Persons: {len(detections)}",
            end=""
        )

except KeyboardInterrupt:
    pass

cam.release()

print("\nStopped")
