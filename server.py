import cv2
import sys
import time
import math
import socket
import numpy as np
from _thread import *
import tensorflow as tf
from yolo.detector import YOLO
from fingertip import Fingertips

graph = tf.get_default_graph()
hand = YOLO(weights='weights/yolo.h5', threshold=0.5)
fingertips = Fingertips(model='vgg', weights='weights/vgg16.h5')

""" Initializing Parameters """
dist = 0
angle = 0
tx, ty = 0.5, 0.5

mode = 'rgb'
if mode == 'gray':
    data_length = 307200
elif mode == 'rgb':
    data_length = 1228800
else:
    mode = 'gray'
    data_length = 307200

""" Setup Communication """
host = "127.0.0.1"
port = 8888
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((host, port))
except:
    print("Connection Failed! :( ")
    sys.exit()

s.listen(2)
print("$$$ Prepared for communication . . . . . ")


def preprocess(byte_data, mode='gray'):
    image = None
    if mode == 'gray':
        image = list(byte_data)
        image = np.asarray(image, dtype=np.uint8)
        image = np.reshape(image, (480, 640))
        image = np.dstack((image * .8, image * .9, image * 1.0))
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.asarray(image, dtype=np.uint8)

    elif mode == 'rgb':
        image = list(byte_data)
        image = np.asarray(image, dtype=np.uint8)
        image = np.reshape(image, (480, 640, 4))
        image = image[:, :, 0:3]

    return image


def client_thread(client, address):
    global graph
    global mode
    global data_length
    global dist
    global angle
    global tx, ty
    print("Connected with {0:10s}:{1:6d} ".format(address[0], address[1]), end='')

    message = '{0},{1},{2},{3}'.format(str(dist), str(angle), str(tx), str(ty))
    client.send(bytes(message.encode('ASCII')))

    while True:
        try:
            data = client.recv(data_length)
            if not data:
                break
            image = preprocess(byte_data=data, mode=mode)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            with graph.as_default():
                tl, br = hand.detect(image)

            if tl or br is not None:
                xmin = tl[0]
                ymin = tl[1]
                xmax = br[0]
                ymax = br[1]

                alpha = 0
                ymin = ymin - alpha if ymin - alpha > 0 else 0
                xmin = xmin - alpha if xmin - alpha > 0 else 0
                cropped_image = image[ymin:ymax + alpha, xmin:xmax + alpha]
                cols, rows, _ = cropped_image.shape
                cropped_image = cv2.resize(cropped_image, (128, 128))
                with graph.as_default():
                    position = fingertips.classify(image=cropped_image)

                for i in range(0, len(position), 2):
                    position[i] = (position[i]) * rows
                    position[i + 1] = (position[i + 1]) * cols

                for i in range(0, len(position), 2):
                    position[i] = (position[i] + tl[0] - alpha)
                    position[i + 1] = (position[i + 1] + tl[1] - alpha)

                image = cv2.rectangle(image, tl, br, (255, 0, 0), 2)
                image = cv2.circle(image, (int(position[0]), int(position[1])), 12, (0, 0, 255), -1)
                image = cv2.circle(image, (int(position[2]), int(position[3])), 12, (0, 255, 0), -1)

                # parameter calculations
                x1, y1, x2, y2 = position[0], position[1], position[2], position[3]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                angle = -math.degrees(math.atan((x1 - x2) / (y1 - y2 + 1e-10)))
                tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
                tx = tx / 640
                ty = ty / 480

            cv2.imshow('', image)
            cv2.waitKey(1)
        except ConnectionResetError:
            print("Connection Lost! :( ")
            break
    client.close()


while True:
    tic = time.time()
    client, address = s.accept()
    start_new_thread(client_thread, (client, address))
    toc = time.time()
    t = toc - tic
    try:
        print('FPS: {0:4.2f} '.format(1 / t), end='')
    except ZeroDivisionError:
        print('FPS: {0:4s} '.format('None'), end='')

    print('Distance: {0:3.2f} '.format(dist))
