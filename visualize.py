import cv2
from preprocess.datagen import label_generator, my_label_gen


def visualize(image, keys):
    """ visualizing the image and the output data """
    image = cv2.circle(image, (int(keys[0]), int(keys[1])), 5, (0, 0, 255), -1)
    image = cv2.circle(image, (int(keys[2]), int(keys[3])), 5, (0, 255, 0), -1)
    cv2.imshow('visual output', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_directory = '../Dataset/Train/'
    label_directory = '../Dataset/label/'
    image_name = 'TI1K_IMAGE_0212.jpg'

    if 'TI1K' in image_name.split('_'):
        image, keypoints = my_label_gen(image_directory=image_directory,
                                        label_directory=label_directory,
                                        image_name=image_name)
    else:
        image, keypoints = label_generator(image_directory=image_directory,
                                           label_directory=label_directory,
                                           image_name=image_name)
    visualize(image, keypoints)
