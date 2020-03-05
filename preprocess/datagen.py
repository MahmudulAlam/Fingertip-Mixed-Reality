import cv2
import numpy as np


def label_generator(image_directory, label_directory, image_name):
    image = cv2.imread(image_directory + image_name)
    image = cv2.resize(image, (416, 416))

    """ Reading corresponding keypoints """
    file = open(label_directory + 'SingleEight.txt')
    lines = file.readlines()
    file.close()
    label = []
    for line in lines:
        line = line.strip().split()
        name = line[0].split('/')[3]
        if image_name == name:
            label = line[1:]
            break

    label = [float(i) * 416 for i in label]
    x1 = int(label[0])
    y1 = int(label[1])
    x2 = int(label[2])
    y2 = int(label[3])

    # cropped image
    alpha = 0
    y1 = y1 - alpha if y1 - alpha > 0 else 0
    x1 = x1 - alpha if x1 - alpha > 0 else 0
    if y1 == 0:
        alpha = 0
    if x1 == 0:
        alpha = 0

    image = image[y1:y2 + alpha, x1:x2 + alpha]
    cols, rows, _ = image.shape
    new_size = 128
    image = cv2.resize(image, (new_size, new_size))
    # keypoints
    tl = [label[0], label[1]]
    keys = label[4:]
    # new keypoints
    label = []
    for i in range(0, len(keys), 4):
        x = (keys[i] - tl[0] + alpha) * new_size / rows
        y = (keys[i + 1] - tl[1] + alpha) * new_size / cols
        label.append(x)
        label.append(y)

    count = 0
    keypoints = []

    keypoints.append(label[count])
    keypoints.append(label[count + 1])
    count = count + 2

    keypoints.append(label[count])
    keypoints.append(label[count + 1])

    keypoints = np.asarray(keypoints)
    return image, keypoints


def my_label_gen(image_directory, label_directory, image_name):
    image = cv2.imread(image_directory + image_name)
    image = cv2.resize(image, (416, 416))

    """ Reading corresponding keypoints """
    file = open(label_directory + 'TI1K.txt')
    lines = file.readlines()
    file.close()
    label = []
    for line in lines:
        line = line.strip().split()
        if image_name == line[0]:
            label = line[1:]

    label = [float(i) * 416 for i in label]
    x1 = int(label[0])
    y1 = int(label[1])
    x2 = int(label[2])
    y2 = int(label[3])

    # cropped image
    alpha = 0
    y1 = y1 - alpha if y1 - alpha > 0 else 0
    x1 = x1 - alpha if x1 - alpha > 0 else 0
    if y1 == 0:
        alpha = 0
    if x1 == 0:
        alpha = 0

    image = image[y1:y2 + alpha, x1:x2 + alpha]
    cols, rows, _ = image.shape
    new_size = 128
    image = cv2.resize(image, (new_size, new_size))
    # keypoints
    tl = [label[0], label[1]]
    keys = label[4:]
    # new keypoints
    label = []
    for i in range(0, len(keys), 2):
        x = (keys[i] - tl[0] + alpha) * new_size / rows
        y = (keys[i + 1] - tl[1] + alpha) * new_size / cols
        label.append(x)
        label.append(y)

    keypoints = []

    keypoints.append(label[0])
    keypoints.append(label[1])
    keypoints.append(label[2])
    keypoints.append(label[3])

    keypoints = np.asarray(keypoints)
    return image, keypoints
