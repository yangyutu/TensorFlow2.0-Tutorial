
# The static graph
## The Static Graph in TensorFlow2.0 as a memorial
## In order to be compatible to the old versions,
## TensorFlow 2.X supports the TensorFlow 1.X styled static graph in the sub-module `tf.compat.v1`.
## This is just for memorial and we do NOT recommend this way.
import tensorflow as tf

g = tf.compat.v1.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(name='x', shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name='y', shape=[], dtype=tf.string)
    z = tf.compat.v1.string_join([x, y], name='join', separator= ' ')

# Executing the graph
with tf.compat.v1.Session(graph = g) as sess:
    print(sess.run(fetches = z, feed_dict = {x:'hello', y:'world'}))

# The Dynamic Graph
# TensorFlow 2.X uses the dynamic graph and Autograph.
# In TensorFlow 1.X, the static graph is impelmented in two steps: defining the graph and executing it in `Session`.
# However, the definition and execution is no more distinguishable for dynamic graph. It executes immediatly after
# definition and that's the reason why it is called "Eager Excution".

x = tf.constant('hello')
y = tf.constant('world')
z = tf.strings.join([x, y], separator=' ')
tf.print(z)

def strjoin(x,y):
    z =  tf.strings.join([x,y],separator = " ")
    tf.print(z)
    return z

result = strjoin(tf.constant("hello"),tf.constant("world"))
print(result)


# create logdic
import datetime
import os
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', stamp)

writer = tf.summary.create_file_writer(logdir)
# start tracing on Autograph
tf.summary.trace_on(graph = True, profiler = True)

# Execute Autograph
result = strjoin('hello', 'world')

# Write the graph info into the log
with writer.as_default():
    tf.summary.trace_export(
        name = "autograph",
        step = 0,
        profiler_outdir=logdir
    )

