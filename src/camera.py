from picamera2 import Picamera2
import cv2


class Camera:
    def __init__(self, width=640, height=480):

        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )

        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        return frame

    def release(self):
        self.picam2.stop()
