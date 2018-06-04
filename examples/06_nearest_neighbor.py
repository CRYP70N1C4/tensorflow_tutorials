import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder(tf.float32, [None, 784])
xte = tf.placeholder(tf.float32, [784])

distance = tf.reduce_sum(tf.abs(xtr - xte), reduction_indices=1)
pred = tf.argmin(distance, 0)
accuracy = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("test {} ,prediction = {} , real class = {}".format(i, np.argmax([Ytr[nn_index]]), np.argmax(Yte[i])))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)

print("accuracy = {}".format(accuracy))
