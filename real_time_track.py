import cv2
import time
import numpy as np
from math import sqrt
from net.vgg16 import model
from darkflow.net.build import TFNet

""" Hand Detection """
options = {'model': 'cfg/tiny-yolo-voc-1c.cfg', 'load': 3250, 'threshold': 0.05, 'gpu': 1.0}
tfnet = TFNet(options)

""" Key points detection """
model = model()
model.summary()
model.load_weights('weights/fingertip_weights/vgg16.h5')


def detect_hand(image):
    """ Hand detection """
    output = tfnet.return_predict(image)
    tl, br = None, None
    for prediction in output:
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])
    return tl, br


def classify(model, image):
    """ Fingertip detection """
    image = np.asarray(image)
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    position = model.predict(image)
    position = position[0]
    return position


""" Kernelized Correlation Filters Tracking """
tracker = cv2.TrackerKCF_create()
cam = cv2.VideoCapture(0)
start = True
mva = 0

while True:
    tic = time.time()
    ret, image = cam.read()

    if ret is False:
        break

    ok, bbox = tracker.update(image)

    if ok is True:
        # print('Tracking . . .')
        alpha_x = 0
        alpha_y = 0
        tl = (int(bbox[0] - alpha_x), int(bbox[1]) - alpha_y)
        br = (int(bbox[0] + bbox[2] + alpha_x), int(bbox[1] + bbox[3]) + alpha_y)

        """ Fingertip detection """
        xmin = int(tl[0])
        ymin = int(tl[1])
        xmax = int(br[0])
        ymax = int(br[1])

        ymin = ymin if ymin > 0 else 0
        xmin = xmin if xmin > 0 else 0

        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape  # heights, width, channel
        cropped_image = cv2.resize(cropped_image, (128, 128))
        position = classify(model=model, image=cropped_image)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        """ Drawing bounding box and fingertips """
        image = cv2.rectangle(image, tl, br, (255, 0, 0), 2, 1)
        image = cv2.circle(image, (int(position[0]), int(position[1])), 12, (0, 0, 255), -1)
        image = cv2.circle(image, (int(position[2]), int(position[3])), 12, (0, 255, 0), -1)

        dist = sqrt((position[0] - position[2]) ** 2 + (position[1] - position[3]) ** 2)

        if start is True:
            mva = dist
            start = False
        else:
            mva = mva * 0.8 + dist * 0.2
        print('Moving Average Distance: {0:4.2f}    '.format(mva), end='')

    else:
        print('Not tracking . . .   ', end='')
        """ Initialize tracking """
        tracker = cv2.TrackerKCF_create()
        ret, image = cam.read()
        """ Hand detection """
        tl, br = detect_hand(image)
        if tl and br is not None:
            """ From rectangular bounding box to square bounding box """
            center_x = int((tl[0] + br[0]) / 2)
            center_y = int((tl[1] + br[1]) / 2)
            image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
            h = br[1] - tl[1]
            w = br[0] - tl[0]

            m = int(max(h, w) / 2)
            x = center_x - m
            y = center_y - m
            m = m + int(m * 0.1)

            bbox = (x, y, 2 * m, 2 * m)
            ok = tracker.init(image, bbox)

    cv2.imshow('Real-time Thumb & Index Detection with Tracking', image)
    if cv2.waitKey(1) & 0xff == 27:
        break

    toc = time.time()
    try:
        t = toc - tic
        FPS = 1 / t
        print('FPS: {0:.2f}'.format(FPS))
    except ZeroDivisionError:
        pass

cam.release()
cv2.destroyAllWindows()
