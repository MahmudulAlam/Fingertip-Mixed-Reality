import cv2
import numpy as np
from yolo.detector import YOLO
from compare.net.deepfinger import deepFinger

hand = YOLO('../../weights/yolo.h5', threshold=0.3)
fingertip = deepFinger()
fingertip.load_weights('../../compare/weights/deepfinger.h5')


def classify(img):
    """ Fingertip detection """
    global fingertip
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    keys = fingertip.predict(img)
    keys = keys[0]
    return keys


user = 1
transformation = 'translation'
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
        cropped_image = cv2.resize(cropped_image, (99, 99))
        position = classify(img=cropped_image)

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

translate_x = []
translate_x_hat = []
translate_y = []
translate_y_hat = []

""" Translation Transformation """
for x1, x2, y1, y2 in zip(xt, xi, yt, yi):
    tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
    tx = tx / 640
    ty = ty / 480
    # transforming normalized value to a corresponding translation value
    # using a linear equation
    tx = 2 * tx - 1.0  # y-axis of the virtual object
    ty = 1 * ty - 0.5  # z-axis of the virtual object
    translate_x.append(tx)
    translate_y.append(ty)

for x1, x2, y1, y2 in zip(xt_hat, xi_hat, yt_hat, yi_hat):
    tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
    tx = tx / 640
    ty = ty / 480
    # transforming normalized value to a corresponding translation value
    # using a linear equation
    tx = 2 * tx - 1.0  # y-axis of the virtual object
    ty = 1 * ty - 0.5  # z-axis of the virtual object
    translate_x_hat.append(tx)
    translate_y_hat.append(ty)

print('')
print('t_x' + str(user) + '_true_1 = ', translate_x, ';')
print('t_x' + str(user) + '_pred_1 = ', translate_x_hat, ';')
print('t_y' + str(user) + '_true_1 = ', translate_y, ';')
print('t_y' + str(user) + '_pred_1 = ', translate_y_hat, ';')
print('')

translate_x = np.asarray(translate_x)
translate_x_hat = np.asarray(translate_x_hat)
translate_y = np.asarray(translate_y)
translate_y_hat = np.asarray(translate_y_hat)

tx_tx_hat = np.mean(np.abs(translate_x - translate_x_hat))
ty_ty_hat = np.mean(np.abs(translate_y - translate_y_hat))
r_tx = np.corrcoef(translate_x, translate_x_hat)
r_ty = np.corrcoef(translate_y, translate_y_hat)

print('{0} & {1} & {2} & {3}'.format(np.round(tx_tx_hat, 4),
                                     np.round(ty_ty_hat, 4),
                                     np.round(r_tx[0][1], 4),
                                     np.round(r_ty[0][1], 4)))
