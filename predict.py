import cv2
from yolo.detector import YOLO
from fingertip import Fingertips

hand = YOLO(weights='weights/yolo.h5', threshold=0.5)
fingertip = Fingertips(model='vgg', weights='weights/vgg16.h5')

image_file = 'data/sample.jpg'
image = cv2.imread(image_file)
tl, br = hand.detect(image=image)

if tl and br is not None:
    alpha_x = 0
    alpha_y = 0
    tl = (tl[0] - alpha_x, tl[1] - alpha_y)
    br = (br[0] + alpha_x, br[1] + alpha_y)

    xmin = int(tl[0])
    ymin = int(tl[1])
    xmax = int(br[0])
    ymax = int(br[1])

    ymin = ymin if ymin > 0 else 0
    xmin = xmin if xmin > 0 else 0

    cropped_image = image[ymin:ymax, xmin:xmax]
    cols, rows, _ = cropped_image.shape
    cropped_image = cv2.resize(cropped_image, (128, 128))
    # cv2.imwrite('cropped_image.jpg', cropped_image)
    position = fingertip.classify(image=cropped_image)

    position[0] = (position[0]) * rows + tl[0]
    position[1] = (position[1]) * cols + tl[1]
    position[2] = (position[2]) * rows + tl[0]
    position[3] = (position[3]) * cols + tl[1]

    """ Drawing bounding box and fingertip """
    image = cv2.rectangle(image, tl, br, (235, 26, 158), 4, 1)
    image = cv2.circle(image, (int(position[0]), int(position[1])), 14, (0, 0, 255), -1)
    image = cv2.circle(image, (int(position[2]), int(position[3])), 14, (0, 255, 0), -1)
    cv2.imshow('predicted_image', image)
    cv2.imwrite('data/' + image_file[:-4] + '_out.jpg', image)
    cv2.waitKey(0)
