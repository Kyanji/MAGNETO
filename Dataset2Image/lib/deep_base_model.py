from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from keras.layers import Dense, Input, Activation, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split  # load MNIST dataset
from keras.layers import Concatenate
from keras.optimizers import SGD
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# from sparse label to categorical
def model_train(x_train, y_train, param):
    # x_train, x_test, y_train, y_test = train_test_split(dataset,
    #                                                     y,
    #                                                     test_size=0.1,
    #                                                     random_state=100)
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # reshape and normalize input images
    x_train = np.array(x_train)
    # x_test = np.array(x_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    # x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    # use functional API to build cnn layers
    inputs = Input(shape=input_shape)
    y1 = Conv2D(filters=param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(inputs)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    # Layer 2
    y2 = Conv2D(filters=param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    # Addition Layer
    y3 = Concatenate()([y1, y2])
    y3 = AveragePooling2D(strides=2, pool_size=2)(y3)
    y3 = Flatten()(y3)
    outputs = Dense(num_labels, activation='softmax')(y3)

    # build the model by supplying inputs/outputs
    model = Model(inputs=inputs, outputs=outputs)
    # network model in text
    model.summary()

    # classifier loss, Adam optimizer, classifier accuracy
    sgd = SGD(lr=param["learning_rate"], momentum=param["momentum"], )
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # train the model with input images and labels
    model.fit(x_train,
              y_train,
              # validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size,
              use_multiprocessing=True)

    # model accuracy on test dataset
    return model


def model_train_with_val(dataset, y, param):
    K.clear_session()
    print(param)
    x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=100)
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # reshape and normalize input images
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    # use functional API to build cnn layers
    inputs = Input(shape=input_shape)
    y1 = Conv2D(filters=param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(inputs)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=2*param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=4*param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(strides=2, pool_size=2)(y1)

    y1 = Conv2D(filters=8*param["filter_size"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    # Layer 2
    y2 = Conv2D(filters=param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=2*param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=4*param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = MaxPooling2D(strides=2, pool_size=2)(y2)

    y2 = Conv2D(filters=8*param["filter_size2"],
                kernel_size=(param["kernel"], param["kernel"]),
                padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    # Addition Layer
    y3 = Concatenate()([y1, y2])
    y3 = AveragePooling2D(strides=2, pool_size=2)(y3)
    y3 = Flatten()(y3)
    outputs = Dense(num_labels, activation='softmax')(y3)

    # build the model by supplying inputs/outputs
    model = Model(inputs=inputs, outputs=outputs)
    # network model in text
    # model.summary()

    sgd = SGD(lr=param["learning_rate"], momentum=param["momentum"])
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # train the model with input images and labels
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size,
              use_multiprocessing=False,
              verbose=1)

    # model accuracy on test dataset
    score = model.evaluate(x_test,
                           y_test,
                           batch_size=batch_size,
                           verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
    print("accuracy:"+str(100.0 * score[1]))

    return 100.0 * score[1]
