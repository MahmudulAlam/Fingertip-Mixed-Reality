from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout


def dtfd():
    input = Input(shape=(99, 99, 3))
    x = ZeroPadding2D((1, 1))(input)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(160, activation='relu')(x)
    x = Dropout(0.5)(x)
    position = Dense(4, activation='linear', name='positional_output')(x)
    model = Model(input=input, outputs=position)
    return model


if __name__ == '__main__':
    model = dtfd()
    model.summary()
