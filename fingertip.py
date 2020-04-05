import numpy as np
from net.vgg16 import model as vgg_model
from net.inception import model as inception_model
from net.xception import model as xception_model
from net.mobilenet import model as mobilenet_model


class Fingertips:
    def __init__(self, model, weights):
        if model is 'vgg':
            self.model = vgg_model()
        elif model is 'inception':
            self.model = inception_model()
        elif model is 'xception':
            self.model = xception_model()
        elif model is 'mobilenet':
            self.model = mobilenet_model()
        else:
            assert False, model + ' does not exist.'
        self.model.load_weights(weights)

    def classify(self, image):
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        position = self.model.predict(image)
        position = position[0]
        return position
