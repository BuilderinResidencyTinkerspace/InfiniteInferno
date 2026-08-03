import cv2
import numpy as np

INPUT_SIZE = 640


def preprocess(frame):
    image = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image
