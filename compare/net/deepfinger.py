from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Concatenate


def deepFinger():
    input = Input(shape=(99, 99, 3))
    x = Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(input)
    x = Conv2D(32, (4, 4), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    y1 = x

    x = Conv2D(96, (2, 2), activation='relu')(x)
    x = Conv2D(96, (2, 2), activation='relu')(x)
    x = ZeroPadding2D((3, 3))(x)
    x = MaxPooling2D((2, 2))(x)

    y2 = x

    y1 = Flatten()(y1)
    y2 = Flatten()(y2)
    y1 = Dense(160, activation='relu')(y1)
    y2 = Dense(160, activation='relu')(y2)
    y = Concatenate()([y1, y2])
    y = Dropout(0.5)(y)
    position = Dense(4, activation='linear', name='positional_output')(y)
    model = Model(input=input, outputs=position)
    return model


if __name__ == '__main__':
    model = deepFinger()
    model.summary()
