import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import model
from preprocess.generator import train_generator, valid_generator


# Custom loss function
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


# Creating the model
model = model()
model.summary()

# Compile
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
model.compile(optimizer=adam, loss=huber_loss)

# Train
epochs = 10
train_gen = train_generator(sample_per_batch=2, batch_number=1690)
val_gen = valid_generator(sample_per_batch=50, batch_number=10)

checkpoints = ModelCheckpoint('weights/weights{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=1690, epochs=epochs, verbose=1, shuffle=True,
                              validation_data=val_gen, validation_steps=10,
                              callbacks=[checkpoints], max_queue_size=100)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
