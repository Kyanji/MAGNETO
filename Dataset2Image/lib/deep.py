import numpy as np
from keras import Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, AveragePooling2D, Add, \
    Concatenate, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split


def CNN_Nature(images, y, param=None):
    print(param)
    x_train, x_test, y_train, y_test = train_test_split(images,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=100)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    image_size = x_train.shape[1]
    image_size2 = x_train.shape[2]

    x_train = np.reshape(x_train, [-1, image_size, image_size2, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size2, 1])

    num_filters = param["filter"]
    num_filters2 = param["filter2"]

    kernel = param["kernel"]

    inputs = Input(shape=(image_size, image_size2, 1))
    print(x_train.shape)
    out = Conv2D(filters=num_filters,
                 kernel_size=(kernel, kernel),
                 padding="same")(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(strides=2, pool_size=2)(out)

    out = Conv2D(filters=2 * num_filters,
                 kernel_size=(kernel, kernel),
                 padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(strides=2, pool_size=2)(out)

    out = Conv2D(filters=4 * num_filters,
                 kernel_size=(kernel, kernel),
                 padding="same")(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # layer 2
    out2 = Conv2D(filters=num_filters2,
                  kernel_size=(kernel, kernel),
                  padding="same")(inputs)
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)
    out2 = MaxPooling2D(strides=2, pool_size=2)(out2)

    out2 = Conv2D(filters=2 * num_filters2,
                  kernel_size=(kernel, kernel),
                  padding="same")(out2)
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)
    out2 = MaxPooling2D(strides=2, pool_size=2)(out2)

    out2 = Conv2D(filters=4 * num_filters2,
                  kernel_size=(kernel, kernel),
                  padding="same")(out2)
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)

    # final layer
    outf = Concatenate()([out, out2])
    out_f = AveragePooling2D(strides=2, pool_size=2)(outf)
    out_f = Flatten()(out_f)
    predictions = Dense(2, activation='softmax')(out_f)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)

    adam = Adam(lr=param["learning_rate"])

    # Compile the model.
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model.
    hist = model.fit(
        x_train,
        y_train,
        epochs=param["epoch"],
        verbose=2,
        validation_data=(x_test, y_test),
        batch_size=param["batch"],
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10),
                   ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    )
    model.load_weights('best_model.h5')

    y_test = np.argmax(y_test, axis=1)

    Y_predicted = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)

    Y_predicted = np.argmax(Y_predicted, axis=1)

    cf = confusion_matrix(y_test, Y_predicted)

    return model, {"balanced_accuracy_val": balanced_accuracy_score(y_test, Y_predicted) * 100, "TN_val": cf[0][0],
                   "FP_val": cf[0][1], "FN_val": cf[1][0], "TP_val": cf[1][1]
                   }


def CNN2(images, y, params=None):
    print(params)
    x_train, x_test, y_train, y_test = train_test_split(images,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=100
                                                        )
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    num_filters = params["filter"]
    num_filters2 = params["filter2"]

    kernel = params["kernel"]

    inputs = Input(shape=(image_size, x_train.shape[2], 1))

    X = Conv2D(32, (2, 2), activation='relu', name='conv0')(inputs)
    X = Dropout(rate=params['dropout1'])(X)
    X = Conv2D(64, (2, 2), activation='relu', name='conv1')(X)
    X = Dropout(rate=params['dropout2'])(X)
    X = Conv2D(128, (1, 2), activation='relu', name='conv2')(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(2, activation='softmax', kernel_initializer='glorot_uniform')(X)

    model = Model(input=inputs, output=X)
    adam = Adam(params["learning_rate"])

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    # Train the model.
    hist = model.fit(
        x_train,
        y_train,
        epochs=params["epoch"],
        verbose=2,
        validation_data=(x_test, y_test),
        batch_size=params["batch"],
        callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10),
                   ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    )
    model.load_weights('best_model.h5')

    y_test = np.argmax(y_test, axis=1)

    Y_predicted = model.predict(x_test, verbose=0, use_multiprocessing=True, workers=12)

    Y_predicted = np.argmax(Y_predicted, axis=1)

    cf = confusion_matrix(y_test, Y_predicted)

    return model, {"balanced_accuracy_val": balanced_accuracy_score(y_test, Y_predicted) * 100, "TN_val": cf[0][0],
                   "FP_val": cf[0][1], "FN_val": cf[1][0], "TP_val": cf[1][1]
                   }

