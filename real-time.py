import cv2
import time
import numpy as np
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


cam = cv2.VideoCapture(0)

while True:
    tic = time.time()
    ret, image = cam.read()

    if ret is False:
        break

    tl, br = detect_hand(image)

    if tl or br is not None:
        xmin, ymin, xmax, ymax = int(tl[0]), int(tl[1]), int(br[0]), int(br[1])
        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape  # heights, width, channel
        resized_image = cv2.resize(cropped_image, (128, 128))
        position = classify(model=model, image=resized_image)

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

    cv2.imshow('Real-time Thumb & Index Detection', image)
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
