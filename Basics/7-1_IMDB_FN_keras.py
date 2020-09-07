import tensorflow as tf
import numpy as np
from tensorflow import keras

max_len = 200
n_words = 10000
dim_embedding = 256
epochs = 20
batch_size = 500

def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words = n_words)
    # pad sequence to the same max_len
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen = max_len)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
    return (X_train, y_train), (X_test, y_test)

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.GlobalAveragePooling1D()) # get the average of the embeddings in a sentence
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

epochs = 30
batch_size = 128
verbose = 1
validation_split = 0.2

(X_train, Y_train), (X_test, Y_test) = load_data()
model = build_model()
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

h = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=verbose, validation_split= validation_split)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_loss, test_acc)

