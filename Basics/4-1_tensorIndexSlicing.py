import tensorflow as tf
import numpy as np
# Indexing and Slicing
## regular slicing
tf.random.set_seed(3)
t = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)

# row 0
tf.print(t[0])

# last row
tf.print(t[-1])

# Row 1 Column 3
tf.print(t[1,3])
tf.print(t[1][3])

# From row 1 to row 3
tf.print(t[1:4,:])
# get a sub matrix
tf.print(tf.slice(t,[1,0],[3,5])) #tf.slice(input,begin_vector,size_vector)
# From row 1 to the last row, and from column 0 to the last but one with an increment of 2
tf.print(t[1:4,0:4:2])

# Variable supports modifying elements through indexing and slicing
x = tf.Variable([[1,2], [3,4]], dtype = tf.float32)
x[1,:].assign(tf.constant([0.0, 0.0]))
tf.print(x)

## advanced slicing
## using tf.gather, tf.gather_nd, and tf.boolean_mask

## 4 classes, 10 students in each class, and 7 courses for each student
scores = tf.random.uniform((4, 10, 7), minval = 0, maxval = 100, dtype=tf.int32)
tf.print(scores)

# get the score of 0, 5, 9 students
p = tf.gather(scores, [0, 5, 9], axis = 1)
tf.print(p)

# get the socre of 0, 5, 9 students at courses 1, 3, 6
q = tf.gather(tf.gather(scores, [0, 5, 9], axis = 1), [1, 3, 6], axis = 2)
tf.print(q)

# Extract all the grades of the 0th student in the 0th class, the 4th student in the 2nd class, and the 6th student in the 3rd class.
# Then length of the parameter indices equals to the number of samples, and the each element of indices is the coordinate of each sample.
s = tf.gather_nd(scores,indices = [(0,0),(2,4),(3,6)])
s

# The function of `tf.gather` and `tf.gather_nd` as shown above could be achieved through `tf.boolean_mask`.
# Get all the grades of the 0, 5, 9 students in each class
p = tf.boolean_mask(scores, [True, False, False, False, False,
                             True, False, False, False, True], axis = 1)
tf.print(p)


# conditional masking
# create a conditional mask
c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
# Find all elements that are less than 0 in the matrix
negative = tf.boolean_mask(c, c < 0)
tf.print(negative)
# a simple way
tf.print(c[c < 0])
# get the mask
mask = c < 0
print(mask)


# Creat new matrix using tf.where and tf.scatter_nd
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
# fill places where c < 0 with np.nan
d = tf.where(c < 0, tf.fill(c.shape, np.nan), c)

# The method where returns all the coordinates that satisfy the condition if there is only one argument
indices = tf.where(c<0)
print(indices)

# Create a new tensor by replacing the value of two tensor elements located at [0,0] [2,1] as 0.
d = c - tf.scatter_nd([[0, 0],[2, 1]],[c[0, 0],c[2, 1]],c.shape)
print(d)



# dimensionality transform

a = tf.random.uniform(shape = [1,3, 3, 2], minval = 0, maxval = 255, dtype=tf.int32)
tf.print(a.shape)
tf.print(a)

# reshape
b = tf.reshape(a, (3, 6))
tf.print(b.shape)
tf.print(b)

# squeeze

a = tf.random.uniform(shape = [1,3, 3, 2], minval = 0, maxval = 255, dtype=tf.int32)
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)

# increase dimensionality
d = tf.expand_dims(s, axis = 0)
print(d.shape)


# combining and splitting
# concat
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])

d = tf.concat([a, b, c], axis = 0)
print(d.shape)
e = tf.concat([a, b, c], axis = 1)
print(e.shape)

# stack, stack will increase dimensionality
f = tf.stack([a, b, c])
print(f.shape)