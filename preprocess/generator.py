import os
import random
import numpy as np
from visualize import visualize
from preprocess.datagen import label_generator, my_label_gen
from preprocess.augmentation import rotation, translate, crop, flip_horizontal, flip_vertical, noise, salt

train_directory = '../Dataset/Train/'
valid_directory = '../Dataset/Valid/'
label_directory = '../Dataset/label/'


def train_generator(sample_per_batch, batch_number):
    """ Generating training data """
    train_image_file = []

    image_files = os.listdir(train_directory)
    train_image_file = train_image_file + image_files
    print('Training Dataset Size: ' + str(len(train_image_file)))

    for i in range(0, 10):
        random.shuffle(train_image_file)

    while True:
        for i in range(0, batch_number - 1):
            start = i * sample_per_batch
            end = (i + 1) * sample_per_batch
            x_batch = []
            y_batch_key = []
            for n in range(start, end):
                image_name = train_image_file[n]

                # image: ndarray keypoints: ndarray but not normalized

                if 'TI1K' in image_name.split('_'):
                    image, keypoints = my_label_gen(image_directory=train_directory,
                                                    label_directory=label_directory,
                                                    image_name=image_name)
                else:
                    image, keypoints = label_generator(image_directory=train_directory,
                                                       label_directory=label_directory,
                                                       image_name=image_name)

                # 1.0 Original Image
                x_batch.append(image)
                y_batch_key.append(keypoints)
                # visualize(image, keypoints)

                """ Augmentation """
                # 2.0 Flip
                im_flip, k_flip = flip_horizontal(image, keypoints)
                x_batch.append(im_flip)
                y_batch_key.append(k_flip)
                # visualize(im_flip, k_flip)

                # 3.0 Original + rotation
                im, k = rotation(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 4.0 Flip + rotation
                im, k = rotation(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 5.0 Original + translate
                im, k = translate(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 6.0 Flip + translate
                im, k = translate(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 7.0 Original + crop
                im, k = crop(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 8.0 Flip + crop
                im, k = crop(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 9.0 Original + noise
                im, k = noise(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 10.0 Flip + noise
                im, k = noise(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 11.0 Original + salt
                im, k = salt(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 12.0 Flip + salt
                im, k = salt(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 13.0 Original + flip vertical
                im, k = flip_vertical(image, keypoints)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 14.0 Flip + flip vertical
                im, k = flip_vertical(im_flip, k_flip)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 15.0 Original + rotate + translate
                im, k = rotation(image, keypoints)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 16.0 Flip + rotate + translate
                im, k = rotation(im_flip, k_flip)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 17.0 Original + noise + translate
                im, k = noise(image, keypoints)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 18.0 Flip + noise + translate
                im, k = noise(im_flip, k_flip)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 19.0 Original + crop + translate
                im, k = crop(image, keypoints)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

                # 20.0 Flip + crop + translate
                im, k = crop(im_flip, k_flip)
                im, k = translate(im, k)
                x_batch.append(im)
                y_batch_key.append(k)
                # visualize(im, k)

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.

            y_batch_key = np.asarray(y_batch_key)
            y_batch_key = y_batch_key.astype('float32')
            y_batch_key = y_batch_key / 128.
            yield (x_batch, y_batch_key)


def valid_generator(sample_per_batch, batch_number):
    """ Generating validation data """

    valid_image_files = os.listdir(valid_directory)

    for i in range(0, 10):
        random.shuffle(valid_image_files)

    while True:
        for i in range(0, batch_number - 1):
            start = i * sample_per_batch
            end = (i + 1) * sample_per_batch
            x_batch = []
            y_batch_key = []
            for n in range(start, end):
                image_name = valid_image_files[n]

                # image: ndarray keypoints: ndarray but not normalized

                if 'TI1K' in image_name.split('_'):
                    image, keypoints = my_label_gen(image_directory=valid_directory,
                                                    label_directory=label_directory,
                                                    image_name=image_name)
                else:
                    image, keypoints = label_generator(image_directory=valid_directory,
                                                       label_directory=label_directory,
                                                       image_name=image_name)

                x_batch.append(image)
                y_batch_key.append(keypoints)

                im_flip, k_flip = flip_horizontal(image, keypoints)
                x_batch.append(im_flip)
                y_batch_key.append(k_flip)

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.

            y_batch_key = np.asarray(y_batch_key)
            y_batch_key = y_batch_key.astype('float32')
            y_batch_key = y_batch_key / 128.
            yield (x_batch, y_batch_key)


if __name__ == '__main__':
    train_directory = '../../Dataset/Train/'
    valid_directory = '../../Dataset/Valid/'
    label_directory = '../../Dataset/label/'

    gen = valid_generator(sample_per_batch=10, batch_number=2)
    x_batch, y_batch = next(gen)
    print(x_batch)
    print(y_batch)
