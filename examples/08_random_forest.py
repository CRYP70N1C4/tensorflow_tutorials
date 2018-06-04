import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)

# Parameters# Param
num_steps = 500  # Total steps to train
batch_size = 1024  # The number of samples per batch
num_classes = 10  # The 10 digits
num_features = 784  # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.int32, [None, num_classes])

hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features,
                                      num_trees=num_trees, max_nodes=max_nodes).fill()

forest_graph = tensor_forest.RandomForestGraphs(hparams)

train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(tf.argmax(Y, 1), tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

sess = tf.train.MonitoredSession()
sess.run(init_vars)

for i in range(1, num_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print("step = {} ,loss = {} , accuracy = {}".format(i, l, acc))

test_x, test_y = mnist.test.images, mnist.test.labels

print("test accuracy = ", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
