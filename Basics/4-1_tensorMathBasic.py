import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3], dtype = tf.float32)
tf.print(a)

b = tf.range(1, 10, delta = 2)
tf.print(b)

c = tf.linspace(0.0, 2 * 3.14, 100)
tf.print(c)

d = tf.zeros([3, 3])
tf.print(d)

e = tf.zeros_like(d, dtype = tf.float32)
tf.print(e)


f = tf.fill((3, 2), 5)
tf.print(f)

# special matrix

I = tf.eye(3, 3)

t = tf.linalg.diag([1, 2, 3])

tf.print(t)

# Random numbers with uniform distribution

tf.random.set_seed(1.0)
a = tf.random.uniform([5], minval = 0, maxval = 10)
tf.print(a)

# Random numbers with normal distribution
b = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
tf.print(b)

