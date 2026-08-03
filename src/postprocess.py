import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

PERSON_CLASS = 0


def postprocess(output, frame_width, frame_height):

    output = np.squeeze(output)
    output = output.T

    boxes = []
    scores = []
    class_ids = []

    x_factor = frame_width / 640
    y_factor = frame_height / 640

    for row in output:

        xc, yc, w, h = row[:4]

        class_scores = row[4:]

        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score < CONFIDENCE_THRESHOLD:
            continue

        if class_id != PERSON_CLASS:
            continue

        left = int((xc - w / 2) * x_factor)
        top = int((yc - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        boxes.append([left, top, width, height])
        scores.append(float(score))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []

    if len(indices) > 0:

        for i in indices.flatten():

            detections.append({
                "box": boxes[i],
                "score": scores[i],
                "class": class_ids[i]
            })

    return detections
