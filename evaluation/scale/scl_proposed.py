import cv2
import numpy as np
from utils.map import map
from yolo.detector import YOLO
from fingertip import Fingertips

hand = YOLO('../../weights/yolo.h5', threshold=0.5)
fingertip = Fingertips(model='vgg', weights='../../weights/vgg16.h5')

user = 1
transformation = 'scale'
f_count = 0

cam = cv2.VideoCapture('../../data/' + transformation + '/' + str(user) + '.mp4')
f = open('../../data/' + transformation + '/' + transformation + '_' + str(user) + '.txt', 'r')
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

height = 480
width = 640

while True:
    ret, image = cam.read()

    if ret is False:
        break

    image = cv2.resize(image, (width, height))
    tl, br = hand.detect(image=image)

    if tl or br is not None:
        line = lines[f_count]
        label = line.strip().split()
        name = label[0]
        label = label[1:]
        label = [float(i) for i in label]

        xt.append(label[4] * width)
        yt.append(label[5] * height)
        xi.append(label[6] * width)
        yi.append(label[7] * height)

        # fingertip detection
        xmin = int(tl[0])
        ymin = int(tl[1])
        xmax = int(br[0])
        ymax = int(br[1])

        ymin = ymin if ymin > 0 else 0
        xmin = xmin if xmin > 0 else 0

        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape
        cropped_image = cv2.resize(cropped_image, (128, 128))
        position = fingertip.classify(image=cropped_image)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        xt_hat.append(position[0])
        yt_hat.append(position[1])
        xi_hat.append(position[2])
        yi_hat.append(position[3])

        """ Drawing bounding box and fingertip """
        # image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
        # image = cv2.circle(image, (int(position[0]), int(position[1])), 10, (0, 0, 255), -1)
        # image = cv2.circle(image, (int(position[2]), int(position[3])), 10, (0, 255, 0), -1)
        # cv2.imshow('', image)
        # cv2.waitKey(0)
        # cv2.imwrite('output/' + image_file, image)

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

""" Scale Transformation """
m, c = map(x1=100, x2=180, y1=.05, y2=.20)
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

print('')
print('d' + str(user) + '_true_0 = ', D.tolist(), ';')
print('d' + str(user) + '_pred_0 = ', D_hat.tolist(), ';')
print('s' + str(user) + '_true_0 = ', scale, ';')
print('s' + str(user) + '_pred_0 = ', scale_hat, ';')
print('')

scale = np.asarray(scale)
scale_hat = np.asarray(scale_hat)
s_s_hat = np.mean(np.abs(scale - scale_hat))

r_d = np.corrcoef(D, D_hat)
r_s = np.corrcoef(scale, scale_hat)

print('{0} & {1} & {2} & {3}'.format(np.round(d_d_hat, 4),
                                     np.format_float_scientific(s_s_hat, unique=False, precision=2),
                                     np.round(r_d[0][1], 4),
                                     np.round(r_s[0][1], 4)))
