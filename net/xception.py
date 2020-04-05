from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import Xception


def model():
    model = Xception(include_top=False, input_shape=(128, 128, 3))
    x = model.output

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    position = Dense(4, activation='sigmoid', name='positional_output')(x)
    model = Model(inputs=model.input, outputs=position)
    return model


if __name__ == '__main__':
    model = model()
    model.summary()
