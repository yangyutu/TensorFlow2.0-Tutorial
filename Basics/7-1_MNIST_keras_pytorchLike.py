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
class MyModel1(tf.keras.Model):
    def __init__(self, n_hidden, n_out):
        super(MyModel1, self).__init__()
        self.dense1 = tf.keras.layers.Dense(n_hidden, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_out, activation='softmax')
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel1(n_hidden, nb_classes)

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

h = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=verbose, validation_split= validation_split)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_loss, test_acc)


# class MyModel2(tf.keras.Model):
#     def __init__(self, n_hidden, n_out):
#         super(MyModel2, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(n_hidden, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(n_out, activation='softmax')
#         self.dropout = tf.keras.layers.Dropout(0.5)
#     def call(self, inputs, training=False):
#         x = self.dense1(inputs)
#         if training:
#             x = self.dropout(x, training=training)
#         return self.dense2(x)
#
# model = MyModel2(n_hidden, nb_classes)
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# h = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=verbose, validation_split= validation_split)
# # evaluate the model
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print(test_loss, test_acc)