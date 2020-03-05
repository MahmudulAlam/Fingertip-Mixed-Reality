import cv2
import sys
import time
import numpy as np
from utils.map import map
from net.vgg16 import model
from darkflow.net.build import TFNet

""" Hand Detection """
options = {'model': 'cfg/tiny-yolo-voc-1c.cfg', 'load': 3250, 'threshold': 0.05, 'gpu': 1.0}
tfnet = TFNet(options)

""" Key points detection """
model = model()
model.summary()
model.load_weights('../weights/fingertip_weights/vgg16.h5')


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

ok = False
f_count = 0
participant_id = 1
image = np.asarray([])
cam = cv2.VideoCapture('Participant 0' + str(participant_id) + '.mp4')

""" Ground truths Annotation """
f = open('Participant 0' + str(participant_id) + '.txt', 'r')
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

width = 640
height = 480

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
        # image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
        # image = cv2.circle(image, (int(position[0]), int(position[1])), 14, (0, 0, 255), -1)
        # image = cv2.circle(image, (int(position[2]), int(position[3])), 14, (0, 255, 0), -1)

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
            image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
            tl_ = tl
            br_ = br
            h = br[1] - tl[1]
            w = br[0] - tl[0]

            m = int(max(h, w) / 2)
            x = center_x - m
            y = center_y - m
            m = m + int(m * 0.1)

            bbox = (x, y, 2 * m, 2 * m)
            ok = tracker.init(image, bbox)
        else:
            print('Tracker not initialized :( ->try lowering threshold value.')
            sys.exit()

    # cv2.imwrite('evaluation/output/Frame_' + str(f_count) + '.jpg', image)
    f_count = f_count + 1
    print('Frame: ', f_count)

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

print("Calculating Pixel Errors")
print("xt yt xi yi")
print("xt - xt_hat: ", np.mean(np.abs(xt - xt_hat)))
print("yt - yt_hat: ", np.mean(np.abs(yt - yt_hat)))
print("xi - xi_hat: ", np.mean(np.abs(xi - xi_hat)))
print("yi - yi_hat: ", np.mean(np.abs(yi - yi_hat)))

print("Calculating Error Between The Fingertips Distance")
D = np.sqrt((xt - xi) ** 2 + (yt - yi) ** 2)
D_hat = np.sqrt((xt_hat - xi_hat) ** 2 + (yt_hat - yi_hat) ** 2)
d_d_hat = np.mean(np.abs(D - D_hat))

scale = []
scale_hat = []
rotate = []
rotate_hat = []
translate = []
translate_hat = []

""" Scale Transformation """
if width is 640:
    m, c = map(100, 180, .05, .20)
    for d in D:
        if d > 180:
            scale.append(0.20)
        elif 100 <= d <= 180:
            scale.append(m * d + c)
        elif d < 100:
            scale.append(0.05)

    for d_hat in D_hat:
        if d_hat > 180:
            scale_hat.append(0.20)
        elif 100 <= d_hat <= 180:
            scale_hat.append(m * d_hat + c)
        elif d_hat < 100:
            scale_hat.append(0.05)

""" Rotation Transformation """
if width is 640:
    m, c = map(100, 180, 0, 45)
    for d in D:
        if d > 180:
            rotate.append(45)
        elif 100 <= d <= 180:
            rotate.append(m * d + c)
        elif d < 100:
            rotate.append(0)

    for d_hat in D_hat:
        if d_hat > 180:
            rotate_hat.append(45)
        elif 100 <= d_hat <= 180:
            rotate_hat.append(m * d_hat + c)
        elif d_hat < 100:
            rotate_hat.append(0)

""" Translation Transformation """
if width is 640:
    m, c = map(100, 180, 0, .5)
    for d in D:
        if d > 180:
            translate.append(0.5)
        elif 100 <= d <= 180:
            translate.append(m * d + c)
        elif d < 100:
            translate.append(0.0)

    for d_hat in D_hat:
        if d_hat > 180:
            translate_hat.append(0.5)
        elif 100 <= d_hat <= 180:
            translate_hat.append(m * d_hat + c)
        elif d_hat < 100:
            translate_hat.append(0)

print('')
print('d' + str(participant_id) + '_true_yolo = ', D.tolist(), ';')
print('d' + str(participant_id) + '_pred_yolo = ', D_hat.tolist(), ';')
print('')
print('s' + str(participant_id) + '_true_yolo = ', scale, ';')
print('s' + str(participant_id) + '_pred_yolo = ', scale_hat, ';')
print('')
print('r' + str(participant_id) + '_true_yolo = ', rotate, ';')
print('r' + str(participant_id) + '_pred_yolo = ', rotate_hat, ';')
print('')
print('t' + str(participant_id) + '_true_yolo = ', translate, ';')
print('t' + str(participant_id) + '_pred_yolo = ', translate_hat, ';')
print('')

scale = np.asarray(scale)
scale_hat = np.asarray(scale_hat)
rotate = np.asarray(rotate)
rotate_hat = np.asarray(rotate_hat)
translate = np.asarray(translate)
translate_hat = np.asarray(translate_hat)

s_s_hat = np.mean(np.abs(scale - scale_hat))
r_r_hat = np.mean(np.abs(rotate - rotate_hat))
t_t_hat = np.mean(np.abs(translate - translate_hat))

r_d = np.corrcoef(D, D_hat)
r_s = np.corrcoef(scale, scale_hat)
r_r = np.corrcoef(rotate, rotate_hat)
r_t = np.corrcoef(translate, translate_hat)

print("correlation: ", r_d[0][1], r_s[0][1], r_r[0][1], r_t[0][1], '\n')
print('{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7}'.format(np.round(d_d_hat, 4),
                                                             np.format_float_scientific(s_s_hat, unique=False,
                                                                                        precision=2),
                                                             np.round(r_r_hat, 4),
                                                             np.format_float_scientific(s_s_hat, unique=False,
                                                                                        precision=2),
                                                             np.round(r_d[0][1], 4),
                                                             np.round(r_s[0][1], 4),
                                                             np.round(r_r[0][1], 4),
                                                             np.round(r_t[0][1], 4)))
