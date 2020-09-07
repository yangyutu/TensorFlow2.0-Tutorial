import numpy as np
import tensorflow as tf

# scalar
i = tf.constant(1) # tf.int32 constant
l = tf.constant(1, dtype = tf.int64) # tf.int64 type constant
f = tf.constant(1.23) # tf.float32 type constant
d = tf.constant(3.14, dtype = tf.double) # tf.double type constant
s = tf.constant("hello world") # tf.string type constant
b = tf.constant(True) # tf.bool type constant

print(tf.int64 == np.int64)
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode)


# matrix

matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(matrix)
print(tf.rank(matrix).numpy())
print(np.ndim(matrix))


# cube

tensor3 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
print(tensor3)
print(tf.rank(tensor3))


# We use tf.cast to change the data type of the tensors
# The method numpy() is for converting the data type from tensors to numpy array
# The method shape is for checking up the size of the tensor

h = tf.constant([123, 456], dtype = tf.int32)
f = tf.cast(h, tf.float32)
h_numpy = h.numpy()
print(h.dtype, f.dtype)
print(h_numpy, type(h_numpy))

# create a tensor from numpy
a = np.array([[1, 2, 3], [4, 5, 6]])
a_tf = tf.convert_to_tensor(a, np.float32)
print(a_tf)


#### variable Tensor
# The trainable parameters in the models are usually defined

# The value of a constant is NOT changeable. Re-assignment creates a new space in the memory.
c = tf.constant([1.0, 2.0])
print(c)
print(id(c))
c = c + tf.constant([2.0, 3.0])
print(c)
print(id(c))

# The value of a variable is changeable through re-assigning methods such as assign, assign_add, etc.
v = tf.Variable([1.0, 2.0], name = "v")
print(v)
print(id(v))
v.assign_add([1.0, 1.0])
print(v)
print(id(v))