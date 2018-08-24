import numpy as np
import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# dataset = dataset.batch(32).repeat()

# val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# val_dataset = val_dataset.batch(32).repeat()


inp = keras.Input(shape=(28, 28, 1)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
x = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(inp)
x = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(rate=0.25)(x)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
# x = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(x)
# x = keras.layers.Conv2D(filters=4, kernel_size=(4, 4), padding='same', activation='relu')(x)
# x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = keras.layers.Dropout(rate=0.25)(x)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=128, activation='relu')(x)
x = keras.layers.Dropout(rate=0.25)(x)
out = keras.layers.Dense(units=10, activation='softmax')(x)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

# The compile step specifies the training configuration.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=20, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)