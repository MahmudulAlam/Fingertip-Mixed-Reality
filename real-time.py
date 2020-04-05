import cv2
import time
from yolo.detector import YOLO
from fingertip import Fingertips

hand = YOLO(weights='weights/yolo.h5', threshold=0.5)
fingertip = Fingertips(model='vgg', weights='weights/vgg16.h5')

cam = cv2.VideoCapture(0)
while True:
    tic = time.time()
    ret, image = cam.read()

    if ret is False:
        print('camera not found! :( ')
        break

    tl, br = hand.detect(image)

    if tl or br is not None:
        xmin, ymin, xmax, ymax = int(tl[0]), int(tl[1]), int(br[0]), int(br[1])
        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape  # heights, width, channel
        resized_image = cv2.resize(cropped_image, (128, 128))
        position = fingertip.classify(image=resized_image)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        """ Drawing bounding box and fingertips """
        image = cv2.rectangle(image, tl, br, (235, 26, 158), 2, 1)
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
