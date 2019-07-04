import cv2
import random
import numpy as np
import imgaug as ia
from preprocess.flag import flag
import imgaug.augmenters as iaa
from preprocess.labelgen import label_generator

f = flag()
size = f.target_size


def augment(image, bbox):
    x = random.randint(-100, 100)
    y = random.randint(-100, 100)
    aug = iaa.Sequential([iaa.Multiply(random.uniform(0.15, 1.5)),
                          iaa.AdditiveGaussianNoise(random.uniform(0, 0.1) * 255),
                          iaa.Affine(translate_px={"x": x, "y": y},
                                     scale=random.uniform(0.35, 1.5),
                                     rotate=random.uniform(-180, 180),
                                     cval=(0, 255))])

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])], shape=image.shape)

    aug = aug.to_deterministic()
    image_aug = aug.augment_image(image)
    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    b = bbs_aug.bounding_boxes
    bbs_aug = [b[0].x1, b[0].y1, b[0].x2, b[0].y2]
    bbs_aug = np.asarray(bbs_aug)

    bbs_aug[0] = bbs_aug[0] if bbs_aug[0] > 0 else 0
    bbs_aug[1] = bbs_aug[1] if bbs_aug[1] > 0 else 0
    bbs_aug[2] = bbs_aug[2] if bbs_aug[2] < size else size
    bbs_aug[3] = bbs_aug[3] if bbs_aug[3] < size else size
    return image_aug, bbs_aug


def augment_flip(image, bbox):
    aug = iaa.Sequential([iaa.Fliplr(1.0)])

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])], shape=image.shape)

    aug = aug.to_deterministic()
    image_aug = aug.augment_image(image)
    image_aug = image_aug.copy()
    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    b = bbs_aug.bounding_boxes
    bbs_aug = [b[0].x1, b[0].y1, b[0].x2, b[0].y2]
    bbs_aug = np.asarray(bbs_aug)

    bbs_aug[0] = bbs_aug[0] if bbs_aug[0] > 0 else 0
    bbs_aug[1] = bbs_aug[1] if bbs_aug[1] > 0 else 0
    bbs_aug[2] = bbs_aug[2] if bbs_aug[2] < size else size
    bbs_aug[3] = bbs_aug[3] if bbs_aug[3] < size else size
    return image_aug, bbs_aug


if __name__ == '__main__':
    folder_directory = '../../Dataset/Train/'
    label_directory = '../../Dataset/label/'
    image_name = 'Avenue_Single_Eight_color_134.jpg'

    file = open(label_directory + 'SingleEight.txt')
    file1 = file.readlines()
    file.close()

    image = cv2.imread(folder_directory + image_name)
    image = cv2.resize(image, (size, size))
    obj_box = label_generator(image_name=image_name, dataset='SingleEight', file=file1)

    image_aug, bbox_aug = augment_flip(image=image, bbox=obj_box)
    image_aug, bbox_aug = augment(image=image_aug, bbox=bbox_aug)
    bbox_aug = [int(b) for b in bbox_aug]

    x1 = bbox_aug[0]
    y1 = bbox_aug[1]
    x2 = bbox_aug[2]
    y2 = bbox_aug[3]

    image_aug = cv2.rectangle(image_aug, (x1, y1), (x2, y2), (255, 15, 15), 3)
    cv2.imshow('', image_aug)
    while True:
        if cv2.waitKey(0) & 0xff == 27:
            break
    cv2.destroyAllWindows()
