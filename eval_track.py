import cv2
import sys
import time
import numpy as np
from model import model
from darkflow.net.build import TFNet

""" Hand Detection """
options = {'model': 'cfg/tiny-yolo-voc-1c.cfg', 'load': 3250, 'threshold': 0.1, 'gpu': 1.0}
tfnet = TFNet(options)

""" Key points detection """
model = model()
model.summary()
model.load_weights('weights/fingertip_weights/Fingertip.h5')


def detect_hand(image):
    """ Hand detection """
    output = tfnet.return_predict(image)
    tl, br = None, None
    for prediction in output:
        # label = prediction['label']
        # confidence = prediction['confidence']
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

participant_id = 1
cam = cv2.VideoCapture('evaluation/Participant 0' + str(participant_id) + '.mp4')
f_count = 0
image = np.asarray([])
ok = False

""" Ground truths Annotation """
f = open('evaluation/Participant 0' + str(participant_id) + '.txt', 'r')
lines = f.readlines()
f.close()

xt = []
yt = []
xi = []
yi = []
xt_hat = []
yt_hat = []
xi_hat = []
yi_hat = []
tl_ = []
br_ = []

height = 480
width = 640

while True:
    tic = time.time()
    if ok is True:
        ret, image = cam.read()
        if ret is False:
            break
        image = cv2.resize(image, (width, height))
    line = lines[f_count]
    line = line.strip().split()
    ok, bbox = tracker.update(image)

    if ok is True:
        alpha_x = 0
        alpha_y = 0
        tl = (int(bbox[0] - alpha_x), int(bbox[1]) - alpha_y)
        br = (int(bbox[0] + bbox[2] + alpha_x), int(bbox[1] + bbox[3]) + alpha_y)

        """ Fingertip detection """
        xmin = int(tl[0])
        ymin = int(tl[1])
        xmax = int(br[0])
        ymax = int(br[1])

        # Correction
        xmin = xmin if xmin > 0 else 0
        ymin = ymin if ymin > 0 else 0
        xmax = xmax if xmax < width else width
        ymax = ymax if ymax < height else height
        tl = (xmin, ymin)
        br = (xmax, ymax)

        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape
        cropped_image = cv2.resize(cropped_image, (128, 128))
        position = classify(model=model, image=cropped_image)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        """ Drawing bounding box and fingertips """
        # image = cv2.rectangle(image, tl_, br_, (255, 0, 0), 4, 1)
        image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
        image = cv2.circle(image, (int(position[0]), int(position[1])), 14, (0, 0, 255), -1)
        image = cv2.circle(image, (int(position[2]), int(position[3])), 14, (0, 255, 0), -1)

        xt.append(float(line[1]) * width)
        yt.append(float(line[2]) * height)
        xi.append(float(line[3]) * width)
        yi.append(float(line[4]) * height)
        xt_hat.append(position[0])
        yt_hat.append(position[1])
        xi_hat.append(position[2])
        yi_hat.append(position[3])

    else:
        """ Initialize tracking """
        tracker = cv2.TrackerKCF_create()
        ret, image = cam.read()
        image = cv2.resize(image, (width, height))
        """ Hand detection """
        tl, br = detect_hand(image)
        if tl and br is not None:
            center_x = int((tl[0] + br[0]) / 2)
            center_y = int((tl[1] + br[1]) / 2)
            # x = center_x - 200
            # y = center_y - 200
            # bbox -> tl_x, tl_y, h, w
            image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
            tl_ = tl
            br_ = br
            h = br[1] - tl[1]
            w = br[0] - tl[0]

            m = int(max(h, w) / 2)
            x = center_x - m
            y = center_y - m
            m = m + int(m * 0.1)

            # x = center_x - int(1.2 * w / 2)
            # y = center_y - int(1.2 * h / 2)
            # bbox = (x, y, int(1.2 * w), int(1.2 * h))
            # bbox = (x, y, 400, 400)
            bbox = (x, y, 2 * m, 2 * m)
            ok = tracker.init(image, bbox)
        else:
            print('Tracker not initialized :( ->try lowering threshold value.')
            sys.exit()

    cv2.imwrite('Eval/output/Frame_' + str(f_count) + '.jpg', image)
    f_count = f_count + 1

cam.release()
cv2.destroyAllWindows()

""" Evaluation """
xt = np.asarray(xt)
yt = np.asarray(yt)
xi = np.asarray(xi)
yi = np.asarray(yi)
xt_hat = np.asarray(xt_hat)
yt_hat = np.asarray(yt_hat)
xi_hat = np.asarray(xi_hat)
yi_hat = np.asarray(yi_hat)

print("##### RESULTS #####")
print("xt yt xi yi")
print("xt - xt_hat: ", np.mean(np.abs(xt - xt_hat)))
print("yt - yt_hat: ", np.mean(np.abs(yt - yt_hat)))
print("xi - xi_hat: ", np.mean(np.abs(xi - xi_hat)))
print("yi - yi_hat: ", np.mean(np.abs(yi - yi_hat)))

print("D D_hat\n")
D = np.sqrt((xt - xi) ** 2 + (yt - yi) ** 2)
D_hat = np.sqrt((xt_hat - xi_hat) ** 2 + (yt_hat - yi_hat) ** 2)
print("D - D_hat: ", np.mean(np.abs(D - D_hat)))
print('d' + str(participant_id) + '_true = ', D.tolist(), ';')
print('d' + str(participant_id) + '_pred = ', D_hat.tolist(), ';')
scale = []
scale_hat = []

if width is 640:
    for d in D:
        if d > 180:
            scale.append(0.20)
        elif 100 <= d <= 180:
            scale.append(0.001875 * d - .1375)
        elif d < 100:
            scale.append(0.05)

    for d_hat in D_hat:
        if d_hat > 180:
            scale_hat.append(0.20)
        elif 100 <= d_hat <= 180:
            scale_hat.append(0.001875 * d_hat - .1375)
        elif d_hat < 100:
            scale_hat.append(0.05)

if width is 320:
    for d in D:
        if d > 90:
            scale.append(0.20)
        elif 50 <= d <= 90:
            scale.append(0.00375 * d - .1375)
        elif d < 50:
            scale.append(0.05)

    for d_hat in D_hat:
        if d_hat > 90:
            scale_hat.append(0.20)
        elif 50 <= d_hat <= 90:
            scale_hat.append(0.00375 * d_hat - .1375)
        elif d_hat < 50:
            scale_hat.append(0.05)

print('s' + str(participant_id) + '_true = ', scale, ';')
print('s' + str(participant_id) + '_pred = ', scale_hat, ';')
print('')
scale = np.asarray(scale)
scale_hat = np.asarray(scale_hat)

print("scale - scale_hat: ", np.mean(np.abs(scale - scale_hat)))
r_d = np.corrcoef(D, D_hat)
r_s = np.corrcoef(scale, scale_hat)
print("correlation: ", r_d, r_s)
