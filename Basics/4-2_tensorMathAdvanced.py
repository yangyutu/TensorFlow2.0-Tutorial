import tensorflow as tf
import numpy as np

# scalar operation

a = tf.constant([[1.0,2],[-3,4.0]])
b = tf.constant([[5.0,6],[7.0,8.0]])
print(a + b)
print(a - b)
print(a * b) # elementwise multiplication
print(a / b) # elementwise division
print(a**2) # elementwise square
print(a**0.5) # elementwise square root
print(a % 3) # elementwise module
print(a // 3)

# get bool array
a = tf.constant([[5.0,6],[7.0,8.0]])
print(a == 5.0)

# get elementwise maximum
a = tf.constant([[1.0,2],[-3,4.0]])
b = tf.constant([[5.0,6],[7.0,8.0]])
tf.print(tf.maximum(a, b))
tf.print(tf.minimum(a, b))

# clip value
x = tf.constant([0.9,-0.8,100.0,-20.0,0.7])
y = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
z = tf.clip_by_norm(x, clip_norm = 3)
print(y)
print(z)

# 2. Vector Operation: manipulate along one specific axis

# reduce
a = tf.range(1, 10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))

# "reduce" along the specific dimension
b = tf.reshape(a,(3,3))
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))

# "reduce" for bool type
p = tf.constant([True,False,False])
q = tf.constant([False,False,True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))

# Cumulative sum
a = tf.range(1,10)
tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))

# Index of max and min values in the arguments
a = tf.range(1,10)
tf.print(tf.argmax(a))
tf.print(tf.argmin(a))

# 3. Matrix Operation
# Matrix operation includes matrix multiply, transpose, inverse, trace, norm, determinant, eigenvalue, decomposition, etc.
# Most of the matrix operations are in the `tf.linalg` except for some popular operations.

## matrix multiplication
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[2,0],[0,2]])
c = a @ b  # Identical to tf.matmul(a,b)
# Matrix transpose
a = tf.constant([[1.0,2],[3,4]])
d = tf.transpose(a)

# Matrix inverse, must be in type of tf.float32 or tf.double
a = tf.constant([[1.0,2],[3.0,4]],dtype = tf.float32)
tf.linalg.inv(a)

# Matrix trace
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.trace(a)

# Matrix norm
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.norm(a)

# Determinant
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.det(a)

# Eigenvalues
a = tf.constant([[1.0,2],[5,4]])
tf.linalg.eigvals(a)

# QR decomposition
a = tf.constant([[1.0,2.0],[3.0,4.0]],dtype = tf.float32)
q, r = tf.linalg.qr(a)
tf.print(q)
tf.print(r)
tf.print(q@r)

# SVD decomposition
a  = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]], dtype = tf.float32)
s,u,v = tf.linalg.svd(a)
tf.print(u,"\n")
tf.print(s,"\n")
tf.print(v,"\n")
tf.print(u@tf.linalg.diag(s)@tf.transpose(v))
