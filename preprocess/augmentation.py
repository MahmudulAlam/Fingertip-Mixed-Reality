import cv2
import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from preprocess.datagen import label_generator, my_label_gen


def rotation(image, keys):
    """ Brightness, rotation, and scaling shear transform """
    aug = iaa.Sequential()
    aug.add(iaa.Multiply(random.uniform(0.20, 1.5)))
    aug.add(iaa.Affine(rotate=random.randint(-180, 180),
                       scale=(random.uniform(0.2, 1.2)),
                       shear=random.randint(-40, 40),
                       cval=(0, 255)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def translate(image, keys):
    """ Translating image """
    aug = iaa.Sequential()
    x = random.randrange(-5, 5) * .01
    y = random.randrange(-5, 5) * .01
    r = random.randrange(-15, 15)
    aug.add(iaa.Affine(translate_percent={"x": x, "y": y}, shear=(r), cval=(0, 255)))
    aug.add(iaa.Multiply((0.20, 1.5)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def crop(image, keys):
    """ Cropping """
    x = random.randint(0, 10)
    y = random.randint(0, 10)
    r = random.randrange(-40, 40)
    aug = iaa.Sequential([iaa.Crop(px=((0, x), (0, y), (0, x), (0, y))), iaa.Affine(shear=r)])
    aug.add(iaa.Multiply((0.20, 1.5)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def flip_horizontal(image, keys):
    """ Flipping """
    aug = iaa.Sequential([iaa.Fliplr(1.0)])
    aug.add(iaa.Multiply(random.uniform(0.20, 1.5)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def flip_vertical(image, keys):
    """ Flipping """
    r = random.randrange(-40, 40)
    aug = iaa.Sequential([iaa.Flipud(1.0), iaa.Affine(shear=r, scale=(random.uniform(.2, 1.2)), cval=(0, 255))])
    aug.add(iaa.Multiply((0.20, 1.5)))
    aug.add(iaa.Affine(rotate=random.randint(-180, 180)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def noise(image, keys):
    """ Adding noise """
    r = random.randint(-10, 10)
    aug = iaa.Sequential([iaa.Affine(translate_px={"x": r, "y": -r}, cval=(0, 255),
                                     rotate=random.randint(-180, 180), shear=r),
                          iaa.AdditiveGaussianNoise(scale=0.1 * 255)])
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


def salt(image, keys):
    """ Adding noise """
    r = random.randint(1, 5) * 0.1
    aug = iaa.Sequential([iaa.Dropout(p=(0, r)), iaa.CoarseDropout(p=0.02, size_percent=0.5),
                          iaa.Salt(0.05)])
    aug.add(iaa.Multiply(random.uniform(0.20, 1.5)))
    aug.add(iaa.Affine(rotate=random.randint(-180, 180)))
    aug.add(iaa.Affine(scale=random.uniform(.2, 1.2), cval=(0, 255)))
    seq_det = aug.to_deterministic()

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]), ia.Keypoint(x=keys[2], y=keys[3])],
                               shape=image.shape)

    image_aug = seq_det.augment_images([image])[0]
    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y]
    keys_aug = np.asarray(keys_aug)
    return image_aug, keys_aug


if __name__ == '__main__':
    folder_directory = '../../Dataset/Train/'
    label_directory = '../../Dataset/label/'
    image_name = 'Avenue_Single_Eight_color_134.jpg'

    if 'TI1K' in image_name.split('_'):
        image, keys = my_label_gen(image_directory=folder_directory,
                                   label_directory=label_directory,
                                   image_name=image_name)
    else:
        image, keys = label_generator(image_directory=folder_directory,
                                      label_directory=label_directory,
                                      image_name=image_name)

    image, k = rotation(image, keys)
    # image, k = translate(image, keys)
    # image, k = crop(image, keys)
    # image, k = flip_horizontal(image, keys)
    # image, k = flip_vertical(image, keys)
    # image, k = noise(image, keys)
    # image, k = salt(image, keys)

    image = cv2.circle(image, (int(k[0]), int(k[1])), 5, (0, 0, 255), -1)
    image = cv2.circle(image, (int(k[2]), int(k[3])), 5, (0, 255, 0), -1)

    cv2.imshow('', image)
    while True:
        if cv2.waitKey(0) & 0xff == 27:
            break
    cv2.destroyAllWindows()
