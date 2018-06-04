import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

a = tf.constant(3)
b = tf.placeholder(tf.int32)
c = tf.multiply(a, b)

maxtrix1 = tf.constant([[3., 4.]])
maxtrix2 = tf.constant([[3.], [4.]])

with tf.Session() as sess:
    print(sess.run(c, feed_dict={b: 7}))
    print(sess.run(tf.sqrt(tf.matmul(maxtrix1, maxtrix2))))
