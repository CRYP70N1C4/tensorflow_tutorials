import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

hello = tf.constant('hello world')

with tf.Session() as sess:
    res = sess.run(hello)

    print(res)
