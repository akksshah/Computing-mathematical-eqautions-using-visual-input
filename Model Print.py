from __future__ import print_function
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python import keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model

num_classes = 10
img_rows, img_cols = 28, 28
seed = 7
np.random.seed(seed)


def train_model(train_x, train_y, test_x, test_y, epoch):
    '''

    :param train_x: train features
    :param train_y: train labels
    :param test_x:  test features
    :param test_y: test labels
    :param epoch: no. of epochs
    :return:

    '''

    batch_size = 128
    if K.image_data_format() == 'channels_first':
        x_train = train_x.reshape(train_x.shape[0], 1, img_rows, img_cols)
        x_test = test_x.reshape(test_x.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)
        x_test = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, train_y,
              batch_size=batch_size,
              epochs=epoch,
              verbose=1,
              validation_data=(x_test, test_y))
    predicted_train_y = model.predict(train_x)
    train_accuracy = (sum(np.argmax(predicted_train_y, axis=1)
                          == np.argmax(train_y, axis=1))/(float(len(train_y))))
    print('Train accuracy : ', train_accuracy)
    predicted_test_y = model.predict(test_x)
    test_accuracy = (sum(np.argmax(predicted_test_y, axis=1)
                         == np.argmax(test_y, axis=1))/(float(len(test_y))))
    print('Test accuracy : ', test_accuracy)
    CNN_accuracy = {'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy, 'epoch': epoch}
    plot_model(model, to_file='model.png')
    return model, CNN_accuracy
