from captcha.image import ImageCaptcha
from PIL import Image
import random
import numpy as np
from keras.models import *
from keras.layers import *

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
    return Image.open(ImageCaptcha(width,height).generate(code))


def gen_test(size=100):
    for _ in range(size):
        code = gen_code()
        gen_image(code).save('test/{}.png'.format(code), 'PNG')


# def encode_label(code):
#     result = np.zeros([length, n_class])
#     for i, s in enumerate(code):
#         result[i][chars.find(s)] = 1
#     return result


def decode_label(array):
    tmp = np.argmax(array, axis=1)
    return "".join(chars[i] for i in tmp)


def gen(batch_size =32):
    batch_x = np.zeros((batch_size,height,width, 3), dtype=np.uint8)
    batch_y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(length)]
    for i in range(batch_size):
        code=gen_code()
        batch_x[i] = gen_image(code)
        for j,ch in enumerate(code) :
            batch_y[j][i] = np.zeros([n_class])
            batch_y[j][i][chars.find(ch)] = 1
    return batch_x,batch_y

def build_model():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(4):
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def train(n_epochs=5,samples_size=51200,batch_size=32):
    model=build_model()
    for _ in range(n_epochs):
        for i in range(int(samples_size/batch_size)):
            batch_x,batch_y = gen(batch_size)
            model.fit(batch_x,batch_y,batch_size=batch_size,validation_data=(batch_x,batch_y),epochs=100)

    model.save('model/captcha.h5')


if __name__ == '__main__':
    train()
    # batch_x,batch_y = gen()
    # print(batch_x.shape)
    # print(batch_y.shape)