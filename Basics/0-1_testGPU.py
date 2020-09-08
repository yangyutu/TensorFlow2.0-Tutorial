import tensorflow as tf

print(tf.test.is_gpu_available())

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))