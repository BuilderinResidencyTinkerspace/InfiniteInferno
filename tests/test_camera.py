from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)

picam2.start_preview(Preview.QT)
picam2.start()

print("Press Ctrl+C to quit")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

picam2.stop()
