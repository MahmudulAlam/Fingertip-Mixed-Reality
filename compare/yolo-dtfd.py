import os
import cv2
import time
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from yolo.detector import YOLO
from compare.net.dtfd import dtfd

hand = YOLO(weights='../weights/yolo.h5', threshold=0.5)
fingertip = dtfd()
fingertip.load_weights('weights/dtfd.h5')


def classify(img):
    """ Fingertip detection """
    global fingertip
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    keys = fingertip.predict(img)
    keys = keys[0]
    return keys


def flip_horizontal(img, keys):
    """ Flipping """
    aug = iaa.Sequential([iaa.Fliplr(1.0)])
    seq_det = aug.to_deterministic()
    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7])], shape=img.shape)

    image_aug = seq_det.augment_images([img])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y]

    return image_aug, keys_aug


image_directory = '../../Dataset/Test/'
label_directory = '../../Dataset/label/'
image_files = os.listdir(image_directory)

""" Ground truth label file for TI1K dataset """
file = open(label_directory + 'TI1K.txt')
lines = file.readlines()
file.close()

""" Ground truth label file for SingleEight dataset """
file = open(label_directory + 'SingleEight.txt')
ego_lines = file.readlines()
file.close()

total_error = np.zeros([1, 4])
avg_hand_detect_time = 0
avg_fingertip_detect_time = 0
avg_time = 0
count = 0
distance_error = []

height = 480
width = 640

for image_file in image_files:
    """ Generating ground truths labels """
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (width, height))
    name = image_file[:-4]
    splits = name.split('_')
    gt = []
    if 'TI1K' in splits:
        label = []
        for line in lines:
            line = line.strip().split()
            if image_file == line[0]:
                label = line[1:]
                break

        label = [float(i) for i in label]
        x1 = label[0] * width
        y1 = label[1] * height
        x2 = label[2] * width
        y2 = label[3] * height
        xt = label[4] * width
        yt = label[5] * height
        xi = label[6] * width
        yi = label[7] * height

        gt = [x1, y1, x2, y2, xt, yt, xi, yi]
        image_flip, gt_flip = flip_horizontal(image, np.asarray(gt))
        image_flip = image_flip.copy()

        """
        [x1, y1, x2, y2, xt, yt, xi, yi] = gt_flip
        image_flip = cv2.rectangle(image_flip, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        image_flip = cv2.circle(image_flip, (int(xt), int(yt)), 14, (0, 0, 255), -1)
        image_flip = cv2.circle(image_flip, (int(xi), int(yi)), 14, (0, 255, 0), -1)
        cv2.imshow('Image', image_flip)
        cv2.waitKey(0)
        """

    else:
        label = []
        for line in ego_lines:
            line = line.strip().split()
            name = line[0].split('/')[3]
            if image_file == name:
                label = line[1:]
                break

        label = [float(i) for i in label]
        x1 = label[0] * width
        y1 = label[1] * height
        x2 = label[2] * width
        y2 = label[3] * height
        xt = label[4] * width
        yt = label[5] * height
        xi = label[8] * width
        yi = label[9] * height

        gt = [x1, y1, x2, y2, xt, yt, xi, yi]
        image_flip, gt_flip = flip_horizontal(image, np.asarray(gt))
        image_flip = image_flip.copy()

        """
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        image = cv2.circle(image, (xt, yt), 14, (0, 0, 255), -1)
        image = cv2.circle(image, (xi, yi), 14, (0, 255, 0), -1)
        cv2.imshow('', image)
        cv2.waitKey(0)
        """

    tic1 = time.time()
    """ Predictions for the test images """
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (width, height))
    tic2 = time.time()
    tl, br = hand.detect(image=image)
    toc2 = time.time()
    avg_hand_detect_time = avg_hand_detect_time + (toc2 - tic2)
    if tl and br is not None:
        # list to tuple
        tl = (tl[0], tl[1])
        br = (br[0], br[1])

        """ Fingertip detection """
        xmin = int(tl[0])
        ymin = int(tl[1])
        xmax = int(br[0])
        ymax = int(br[1])

        ymin = ymin if ymin > 0 else 0
        xmin = xmin if xmin > 0 else 0

        cropped_image = image[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape
        cropped_image = cv2.resize(cropped_image, (99, 99))
        tic3 = time.time()
        position = classify(img=cropped_image)
        toc3 = time.time()
        avg_fingertip_detect_time = avg_fingertip_detect_time + (toc3 - tic3)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        pr = [tl[0], tl[1], br[0], br[1], position[0], position[1], position[2], position[3]]

        """ Drawing bounding box and fingertip """
        image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
        image = cv2.circle(image, (int(position[0]), int(position[1])), 10, (0, 0, 255), -1)
        image = cv2.circle(image, (int(position[2]), int(position[3])), 10, (0, 255, 0), -1)
        cv2.imwrite('output/' + image_file, image)

        # Calculating error for fingertips only
        gt = np.asarray(gt[4:])
        pr = np.asarray(pr[4:])
        abs_err = abs(gt - pr)
        total_error = total_error + abs_err
        D = np.sqrt((gt[0] - gt[2]) ** 2 + (gt[1] - gt[3]) ** 2)
        D_hat = np.sqrt((pr[0] - pr[2]) ** 2 + (pr[1] - pr[3]) ** 2)
        distance_error.append(abs(D - D_hat))
        count = count + 1

    print('Detected Image: {0}'.format(count))

    """ Predictions for the flipped test images """
    tic2 = time.time()
    tl, br = hand.detect(image=image_flip)
    toc2 = time.time()
    avg_hand_detect_time = avg_hand_detect_time + (toc2 - tic2)
    if tl and br is not None:
        # list to tuple
        tl = (tl[0], tl[1])
        br = (br[0], br[1])

        """ Fingertip detection """
        xmin = int(tl[0])
        ymin = int(tl[1])
        xmax = int(br[0])
        ymax = int(br[1])

        ymin = ymin if ymin > 0 else 0
        xmin = xmin if xmin > 0 else 0

        cropped_image = image_flip[ymin:ymax, xmin:xmax]
        cols, rows, _ = cropped_image.shape
        cropped_image = cv2.resize(cropped_image, (99, 99))
        tic3 = time.time()
        position = classify(img=cropped_image)
        toc3 = time.time()
        avg_fingertip_detect_time = avg_fingertip_detect_time + (toc3 - tic3)

        for i in range(0, len(position), 2):
            position[i] = (position[i]) * rows
            position[i + 1] = (position[i + 1]) * cols

        for i in range(0, len(position), 2):
            position[i] = (position[i] + tl[0])
            position[i + 1] = (position[i + 1] + tl[1])

        pr = [tl[0], tl[1], br[0], br[1], position[0], position[1], position[2], position[3]]

        """ Drawing bounding box and fingertip """
        image_flip = cv2.rectangle(image_flip, tl, br, (255, 0, 0), 4, 1)
        image_flip = cv2.circle(image_flip, (int(position[0]), int(position[1])), 10, (0, 0, 255), -1)
        image_flip = cv2.circle(image_flip, (int(position[2]), int(position[3])), 10, (0, 255, 0), -1)
        cv2.imwrite('output/' + image_file[:-4] + '_flip.jpg', image_flip)

        # Calculating error for fingertips only
        gt_flip = np.asarray(gt_flip[4:])
        pr = np.asarray(pr[4:])
        abs_err = abs(gt_flip - pr)
        total_error = total_error + abs_err
        D = np.sqrt((gt[0] - gt[2]) ** 2 + (gt[1] - gt[3]) ** 2)
        D_hat = np.sqrt((pr[0] - pr[2]) ** 2 + (pr[1] - pr[3]) ** 2)
        distance_error.append(abs(D - D_hat))
        count = count + 1
    print('Detected Image: {0}'.format(count))
    toc1 = time.time()
    avg_time = avg_time + (toc1 - tic1)

er = total_error / count
er = er[0]
er = np.round(er, 4)
distance_error = np.array(distance_error)
distance_error = np.mean(distance_error)
distance_error = np.round(distance_error, 4)
print('Total Detected Image: {0}'.format(count))
print('Total Detected Image: {0}'.format(count))
print('Pixel errors: xt = {0}, yt = {1}, xi = {2}, yi = {3}, D-D_hat = {4}'.format(er[0], er[1],
                                                                                   er[2], er[3],
                                                                                   distance_error))

avg_time = avg_time / 1000
avg_hand_detect_time = avg_hand_detect_time / count
avg_fingertip_detect_time = avg_fingertip_detect_time / count

print('Average execution time: {0:1.5f} ms'.format(avg_time * 1000))
print('Average hand detection time: {0:1.5f} ms'.format(avg_hand_detect_time * 1000))
print('Average fingertip detection time: {0:1.5f} ms'.format(avg_fingertip_detect_time * 1000))

print('{0} & {1} & {2} & {3} & {4}'.format(er[0], er[1], er[2], er[3], distance_error))
