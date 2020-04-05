import cv2
import numpy as np
from yolo.darknet import model as yolo_model
from yolo.preprocess.yolo_flag import Flag as yoloFlag


class YOLO:
    def __init__(self, weights, threshold):
        self.f = yoloFlag()
        self.model = yolo_model()
        self.threshold = threshold
        self.model.load_weights(weights)

    def detect(self, image):
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.f.target_size, self.f.target_size)) / 255.0
        image = np.expand_dims(image, axis=0)
        yolo_out = self.model.predict(image)
        yolo_out = yolo_out[0]

        grid_pred = yolo_out[:, :, 0]
        i, j = np.squeeze(np.where(grid_pred == np.amax(grid_pred)))

        if grid_pred[i, j] >= self.threshold:
            bbox = yolo_out[i, j, 1:]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # size conversion
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            return (x1, y1), (x2, y2)
        else:
            return None, None
