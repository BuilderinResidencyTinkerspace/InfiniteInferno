import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from camera import Camera
from preprocess import preprocess
from detector import Detector
from postprocess import postprocess

cam = Camera()

detector = Detector()

print("Running detector...")

while True:

    frame = cam.read()

    tensor = preprocess(frame)

    output = detector.infer(tensor)

    detections = postprocess(
        output,
        frame.shape[1],
        frame.shape[0]
    )

    print(detections)

    break

cam.release()
