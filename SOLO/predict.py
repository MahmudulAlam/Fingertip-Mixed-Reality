import cv2
import numpy as np
from solo import hand_model
from scipy import ndimage
from visualize import visualize

model = hand_model()
model.load_weights('weights/solo.h5')

image = cv2.imread('IMAGES/Image (1).jpg')
image = cv2.resize(image, (416, 416))
img = image / 255.0
img = np.expand_dims(img, axis=0)
grid_output = model.predict(img)
grid_output = grid_output[0]
grid_output = (grid_output > 0.5).astype(int)
print(grid_output)
blob, nBlob = ndimage.label(grid_output)
try:
    biggest_blob = np.bincount(blob.flat)[1:].argmax() + 1
    grid_output = (blob == biggest_blob).astype(int)
except ValueError:
    pass
visualize(image=image, output=grid_output)
