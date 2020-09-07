
# Automatic differentiation

import tensorflow as tf
import numpy as np

# Calculate the derivative of f(x) = a*x**2 + b*x + c
x = tf.Variable(0.0, name='x', dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c
# gradient at x = 0
print(y)
print(x)
dy_dx = tape.gradient(y, x)
print(dy_dx)


# Use watch to calculate derivatives of the constant tensor

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_da)
print(dy_dc)


# Calculate the second order derivative
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape1.gradient(y, x)
dy2_dx2 = tape2.gradient(dy_dx, x)
print(dy2_dx2)


### Calculate the minimal value through the gradient tape and optimizer

x = tf.constant(0.0, name='x', dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

tf.print('y = ', y, '; x = ', x)
