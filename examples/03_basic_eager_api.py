import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)

print(a + b)
