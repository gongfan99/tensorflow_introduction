import numpy as np
import scipy as sp
import h5py
import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# add depth to the input images so that Conv2D layer can consume. depth = 1.
x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

inp = keras.Input(shape=(28, 28, 1)) # depth goes last in TensorFlow back-end (first in Theano)
inpBatchNorm = keras.layers.BatchNormalization()(inp)

outs = []
for i in range(2):
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    x = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(l=0.0001), activation='relu')(inpBatchNorm)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(l=0.0001), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(rate=0.25)(x)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=128, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(l=0.0001), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(rate=0.25)(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)
    outs.append(x)

out = keras.layers.average(outs)
model = keras.Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

# The compile step specifies the training configuration.
# sparse_categorical_crossentropy is used because the targets (or called labels) are intergers.
# if targets are one-hot encoded, then categorical_crossentropy should be used.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train[:54000])

model.fit_generator(datagen.flow(x_train[:54000], y_train[:54000], batch_size=54), steps_per_epoch=54000/54, epochs=2, validation_data=(x_train[54000:], y_train[54000:]), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

loss, acc = model.evaluate(x_test, y_test)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))