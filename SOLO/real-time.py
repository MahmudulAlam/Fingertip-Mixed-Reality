import cv2
import numpy as np
from solo import hand_model
from preprocess.flag import flag

model = hand_model()
model.load_weights('weights/solo.h5')

grid_size = flag().grid_size
cam = cv2.VideoCapture(0)

while True:
    ret, ori_image = cam.read()
    height, width, _ = ori_image.shape

    if ret is False:
        break

    image = cv2.resize(ori_image, (416, 416))
    img = image / 255.0
    img = np.expand_dims(img, axis=0)
    grid_output = model.predict(img)
    grid_output = grid_output[0]
    output = (grid_output > 0.6).astype(int)

    """ Finding bounding box """
    prediction = np.where(output > 0.5)
    row_wise = prediction[0]
    col_wise = prediction[1]
    try:
        x1 = min(col_wise) * grid_size
        y1 = min(row_wise) * grid_size
        x2 = (max(col_wise) + 1) * grid_size
        y2 = (max(row_wise) + 1) * grid_size
        """ size conversion """
        x1 = int(x1 / 416 * width)
        y1 = int(y1 / 416 * height)
        x2 = int(x2 / 416 * width)
        y2 = int(y2 / 416 * height)
        ori_image = cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except ValueError:
        print('NO Hand Detected')

    cv2.imshow('WebCam', ori_image)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
