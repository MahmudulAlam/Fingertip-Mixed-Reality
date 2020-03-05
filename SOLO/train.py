import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from solo import hand_model
from generator import train_generator, valid_generator


def loss_function(y_true, y_pred):
    a = tf.clip_by_value(y_pred, 1e-20, 1)
    b = tf.clip_by_value(tf.subtract(1.0, y_pred), 1e-20, 1)
    cross_entropy = - tf.multiply(y_true, tf.log(a)) - tf.multiply(tf.subtract(1.0, y_true), tf.log(b))
    cross_entropy = tf.reduce_mean(cross_entropy, 1)
    cross_entropy = tf.reduce_mean(cross_entropy, 0)
    loss = tf.reduce_mean(cross_entropy)
    return loss


# create model
model = hand_model()
model.summary()

# compile
adam = Adam(lr=1e-5)
metrics = {'output': loss_function}
model.compile(optimizer=adam, loss=loss_function, metrics=metrics)

# train
epochs = 10
train_gen = train_generator(steps_per_epoch=845, sample_per_batch=4)
val_gen = valid_generator(steps_per_epoch=50, sample_per_batch=10)
checkpoints = ModelCheckpoint('weights/hand_weights{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=845, epochs=epochs, verbose=1,
                              validation_data=val_gen, validation_steps=25,
                              shuffle=True, callbacks=[checkpoints], max_queue_size=50)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
