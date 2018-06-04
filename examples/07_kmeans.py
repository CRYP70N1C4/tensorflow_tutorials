import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import KMeans
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)

full_data_x = mnist.train.images

num_steps = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 784

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) \
    = kmeans.training_graph()

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(init_op, feed_dict={X: full_data_x})
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})

        if i % 10 == 0:
            print("step {} ,avg distance {} ".format(i, d))

    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = mnist.test.images, mnist.test.labels

    print('test accuracy : ', sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
