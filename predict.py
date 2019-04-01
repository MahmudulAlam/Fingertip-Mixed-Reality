import cv2
import numpy as np
from model import model
from darkflow.net.build import TFNet

options = {'model': 'cfg/tiny-yolo-voc-1c.cfg', 'load': 3250, 'threshold': 0.1, 'gpu': 1.0}
tfnet = TFNet(options)

model = model()
model.summary()
model.load_weights('weights/fingertip_weights/Fingertip.h5')


def detect_hand(image):
    """ Hand detection """
    output = tfnet.return_predict(image)
    print(output)
    tl, br = None, None
    for prediction in output:
        # label = prediction['label']
        # confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])
    return tl, br


def classify(model, image):
    """ Fingertip detection """
    image = np.asarray(image)
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    position = model.predict(image)
    position = position[0]
    return position


image_directory = '../Dataset/Test/'
image_file = 'BasketballField_Single_Eight_color_155.jpg'

image = cv2.imread(image_directory + image_file)
tl, br = detect_hand(image=image)

if tl and br is not None:
    alpha_x = 0
    alpha_y = 0
    tl = (tl[0] - alpha_x, tl[1] - alpha_y)
    br = (br[0] + alpha_x, br[1] + alpha_y)

    """ Fingertip detection """
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
    position = classify(model=model, image=cropped_image)

    position[0] = (position[0]) * rows + tl[0]
    position[1] = (position[1]) * cols + tl[1]
    position[2] = (position[2]) * rows + tl[0]
    position[3] = (position[3]) * cols + tl[1]

    """ Drawing bounding box and fingertip """
    image = cv2.rectangle(image, tl, br, (255, 0, 0), 4, 1)
    image = cv2.circle(image, (int(position[0]), int(position[1])), 14, (0, 0, 255), -1)
    image = cv2.circle(image, (int(position[2]), int(position[3])), 14, (0, 255, 0), -1)
    cv2.imshow('predicted_image', image)
    cv2.waitKey(0)
    # cv2.imwrite(image_file, image)
