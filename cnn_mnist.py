from __future__ import print_function

import argparse
import os
from scipy import ndimage
from scipy import misc

import keras
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def train(args):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    model.save('cifar_cnn.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def load_image(full_path):
    img = misc.imread(full_path, mode='L')
    resized = misc.imresize(img, (img_rows, img_cols))
    return 255 - resized


def test(args):
    if args.snapshot is None:
        raise Exception("model snapshot is not specified for test")

    model = keras.models.load_model(args.snapshot)

    # scanning directory for test images
    test_files = os.listdir(args.test_dir)
    for test_file in test_files:
        full_path = os.path.join(args.test_dir, test_file)
        x = load_image(full_path)
        prediction = model.predict(x.reshape(1, img_rows, img_cols, 1), verbose=0)
        print("%s: pediction=%d, probs=%s" % (test_file, np.argmax(prediction[0]), str(prediction[0])))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train CNN on MNIST.')
    parser.add_argument("-snapshot", type=str, default=None, help="model to use at start of training")
    parser.add_argument("-train", type=str, default=False, help="do training")
    parser.add_argument("-test_dir", type=str, default=None, help="directory for test images")

    _args = parser.parse_args()
    if bool(_args.train):
        train(_args)
    else:
        test(_args)
