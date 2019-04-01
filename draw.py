import cv2

# TI1K_IMAGE_0164
# TI1K_IMAGE_0130
# TI1K_IMAGE_0140

""" reading the image """
image_name = 'TI1K_IMAGE_0140.jpg'
image = cv2.imread('../Dataset/Train/' + image_name)

""" reading the annotation """
file = open('../Dataset/label/TI1K.txt')
lines = file.readlines()
file.close()
label = []
for line in lines:
    line = line.strip().split()
    if image_name == line[0]:
        label = line[1:]
        break

""" conversion from normalized to actual value """
label = [float(i) for i in label]
print(label)
x1 = int(label[0] * 640)
y1 = int(label[1] * 480)
x2 = int(label[2] * 640)
y2 = int(label[3] * 480)

tlx = int(label[4] * 640)
tly = int(label[5] * 480)
brx = int(label[6] * 640)
bry = int(label[7] * 480)

""" drawing ground truth annotation """
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
cv2.circle(image, (tlx, tly), 14, (0, 0, 255), -1)
cv2.circle(image, (brx, bry), 14, (0, 255, 0), -1)
cv2.imshow('image', image)
# cv2.imwrite('train.jpg', image)
cv2.waitKey(0)
