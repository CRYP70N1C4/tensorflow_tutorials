from captcha.image import ImageCaptcha
from PIL import Image
import random
import tensorflow as tf
import numpy as np

length = 4
chars = '023456789ABCDEFGHKMNPQRSTUVWXYZ'
width = length * 40
height = 80
n_class = len(chars)


def gen_code():
    code = ''
    for _ in range(length):
        code = code + chars[random.randint(0, n_class - 1)]
    return code


def gen_image(code):
    return Image.open(ImageCaptcha(width, height).generate(code))


def gen_test(size=100):
    for _ in range(size):
        code = gen_code()
        gen_image(code).save('test/{}.png'.format(code), 'PNG')


def decode_label(array):
    tmp = np.argmax(array, axis=1)
    return "".join(chars[i] for i in tmp)


def gen(batch_size=32):
    batch_x = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    batch_y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(length)]
    for i in range(batch_size):
        code = gen_code()
        batch_x[i] = gen_image(code)
        for j, ch in enumerate(code):
            batch_y[j][i] = np.zeros([n_class])
            batch_y[j][i][chars.find(ch)] = 1
    return batch_x, batch_y


def conv_net(x):
    for i in range(4):
        x = tf.layers.conv2d(x, 32 * 2 ** i,3,3,padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32 * 2 ** i,3,3,padding='SAME', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2,2,padding='SAME')

    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, 0.25)
    x = [tf.layers.dense(x, n_class, activation=tf.nn.softmax, name='c%d' % (i + 1)) for i in range(4)]
    return x


def train(batch_size=100, training_epochs=25, learning_rate=0.01, display_step=10):
    X = tf.placeholder(tf.float32, [None, height,width,3])
    Y = tf.placeholder(tf.float32, [length,None, n_class])
    logits = conv_net(X)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.argmax(Y,axis=1)
    ))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            batch_x, batch_y = gen(batch_size)
            loss,y=sess.run([loss_op,Y],feed_dict={X: batch_x, Y: batch_y})
            print(loss.shape)
            print(y.shape)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if (epoch + 1) % display_step == 0:
                print(0)


if __name__ == '__main__':
    train()
    # batch_x,batch_y = gen()
    # print(batch_x.shape)
    # print(batch_y.shape)
