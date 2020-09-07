import tensorflow as tf
import numpy as np
from tensorflow import keras

epochs = 30
batch_size = 128
verbose = 1
nb_classes = 10
n_hidden = 128
validation_split = 0.2

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

reshape = 784
X_train = X_train.reshape(60000, reshape).astype('float32')
X_test = X_test.reshape(10000, reshape).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)
print(X_test.shape)

# one hot representation of the labels
Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, nb_classes)

# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(reshape,)))
model.add(tf.keras.layers.Dense(n_hidden, activation='relu'))
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

h = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=verbose, validation_split= validation_split)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_loss, test_acc)


# build the model with dropout
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.Input(shape=(reshape,)))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.Dense(n_hidden, activation='relu'))
model2.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

model2.summary()
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

h = model2.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=verbose, validation_split= validation_split)

# evaluate the model
test_loss, test_acc = model2.evaluate(X_test, Y_test)
print(test_loss, test_acc)