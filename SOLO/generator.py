import os
import cv2
import random
import numpy as np
from preprocess.flag import flag
from visualize import visualize
from preprocess.labelgen import label_generator, grid_generator
from preprocess.augmentation import augment, augment_flip

f = flag()
target_size = f.target_size
dataset_directory = '../Dataset/Train/'
label_directory = '../Dataset/label/'

file = open(label_directory + 'SingleEight.txt')
file1 = file.readlines()
file.close()

file = open(label_directory + 'TI1K.txt')
file2 = file.readlines()
file.close()


def train_generator(steps_per_epoch, sample_per_batch):
    train_image_files = os.listdir(dataset_directory)

    for i in range(0, 15):
        random.shuffle(train_image_files)

    print('Training Dataset Size: ', len(train_image_files))

    while True:
        for i in range(0, steps_per_epoch):
            x_batch = []
            y_batch = []
            start = i * sample_per_batch
            for n in range(start, start + sample_per_batch):
                image_name = train_image_files[n]
                image = cv2.imread(dataset_directory + image_name)
                image = cv2.resize(image, (target_size, target_size))
                if 'Single' in image_name.split('_'):
                    obj_box = label_generator(image_name=image_name, file=file1, dataset='SingleEight')
                else:
                    obj_box = label_generator(image_name=image_name, file=file2, dataset='TI1K')

                # 01: original image
                x_batch.append(image)
                output = grid_generator(obj_box=obj_box)
                y_batch.append(output)
                # visualize(image=image, output=output)

                # 02: flip image
                image_flip, bbox_flip = augment_flip(image=image, bbox=obj_box)
                x_batch.append(image_flip)
                y_batch.append(grid_generator(obj_box=bbox_flip))

                # 03: original image + augment
                image_aug, bbox_aug = augment(image=image, bbox=obj_box)
                x_batch.append(image_aug)
                output = grid_generator(obj_box=bbox_aug)
                y_batch.append(output)
                # visualize(image=image_aug, output=output)

                # 04: flip image + augment
                image_aug, bbox_aug = augment(image=image_flip, bbox=bbox_flip)
                x_batch.append(image_aug)
                output = grid_generator(obj_box=bbox_aug)
                y_batch.append(output)
                # visualize(image=image_aug, output=output)

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.0
            y_batch = np.asarray(y_batch)

            yield (x_batch, y_batch)


def valid_generator(steps_per_epoch, sample_per_batch):
    dataset_directory = '../Dataset/Valid/'
    train_image_files = os.listdir(dataset_directory)

    for i in range(0, 10):
        random.shuffle(train_image_files)

    while True:
        for i in range(0, steps_per_epoch):
            x_batch = []
            y_batch = []
            start = i * sample_per_batch
            for n in range(start, start + sample_per_batch):
                image_name = train_image_files[n]
                image = cv2.imread(dataset_directory + image_name)
                image = cv2.resize(image, (416, 416))
                if 'Single' in image_name.split('_'):
                    obj_box = label_generator(image_name=image_name, file=file1, dataset='SingleEight')
                else:
                    obj_box = label_generator(image_name=image_name, file=file2, dataset='TI1K')

                # 01: original image
                x_batch.append(image)
                output = grid_generator(obj_box=obj_box)
                y_batch.append(output)
                # visualize(image=image, output=output)

                # 02: flip image
                image_flip, bbox_flip = augment_flip(image=image, bbox=obj_box)
                x_batch.append(image_flip)
                y_batch.append(grid_generator(obj_box=bbox_flip))

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.0
            y_batch = np.asarray(y_batch)

            yield (x_batch, y_batch)


if __name__ == '__main__':
    gen = train_generator(steps_per_epoch=10, sample_per_batch=100)
    x_batch, y_batch = next(gen)
    print(x_batch)
    print(y_batch)
