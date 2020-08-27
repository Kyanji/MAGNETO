from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe, hp
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, concatenate, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, ZeroPadding2D, Activation, Add, AveragePooling2D
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras import callbacks



import pandas as pd
np.random.seed(12)



from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam

#import global_config
from time import perf_counter




def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    pathDataset = 'dataset/UNSW_NB15/'
    path = 'Train'
    pathTest = 'Test_UNSW_NB15'


    train = pd.read_csv(pathDataset + path + ".csv")
    test = pd.read_csv(pathDataset + pathTest + ".csv")
    clsT=' classification.'
    train_normal= train
    test_normal = test
    #train_anormal = train[(train[clsT] == 0)]
    #train_normal = train[(train[clsT] == 1)]
    #test_normal = test[(test[clsT] == 1)]
    #print(train_normal.shape)
    #print(test_normal.shape)
    #test_anormal = test[(test[clsT] == 0)]

    clssList = train.columns.values
    target = [i for i in clssList if i.startswith(clsT)]
    # target = [i for i in clssList if i.startswith(self._clsTrain)]

    # remove label from dataset to create Y ds
    y_train = train_normal[target]
    y_test = test_normal[target]
    # remove label from dataset
    x_train = train_normal.drop(target, axis=1)
    x_train = x_train.values
    x_test = test_normal.drop(target, axis=1)
    x_test = x_test.values


    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

def getBatchSize(p, bs):
        return bs[p]


def Autoencoder(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1],)
    input2 = Input(input_shape)


    # encoder_layer
    # Dropoout?
    #  input1 = Dropout(.2)(input)
    encoded = Dense(128, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(input2)
    encoded = Dense(64, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2')(encoded)
    encoded= Dropout(0)(encoded)
    decoded = Dense(128, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(x_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)


    model = Model(inputs=input2, outputs=decoded)
    model.summary()

    adam=Adam(lr=0.001)
    model.compile(loss='mse', metrics=['acc'],
                  optimizer=adam)
    model.summary()
    model.save("AUTOENCODERUNSW.h5")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True),
    ]

    history= model.fit(x_train, x_train,
                      batch_size={{choice([32,64, 128,256,512])}},
                      epochs=150,
                      verbose=2,
                      validation_split=0.2,callbacks=callbacks_list)


    # get the highest validation accuracy of the training epochs
    score = np.amin(history.history['val_loss'])
    print('Best validation loss of epoch:', score)


    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print('Score',score)


    print('Best score',global_config.best_score)




    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']), 'n_params': model.count_params(), 'model': global_config.best_model}







if __name__ == '__main__':

    x_train, y_train, x_test, y_test=data()
    a=Autoencoder(x_train, y_train, x_test, y_test)
    bs=[32,64,128, 256, 512]
    lr=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                             0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                             0.01
                             ]
    best_run, best_model = optim.minimize(model=Autoencoder,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=trials)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(global_config.best_model.evaluate(X_test, X_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    outfile = open('AutoencoderN.csv', 'w')
    outfile.write("\nHyperopt trials")

    outfile.write("\ntid , loss , learning_rate , Dropout , batch_size")
    for trial in trials.trials:
        #outfile.write(str(trial))
        outfile.write("\n%s , %f , %f , %s , %s" % (trial['tid'],
                                                        trial['result']['loss'],
                                                        trial['misc']['vals']['lr'][0],
                                                        trial['misc']['vals']['Dropout'],
                                                       getBatchSize(trial['misc']['vals']['batch_size'][0] , bs)
                                                        ))

       # outfile.write(str(trial))



    outfile.write('\nBest model:\n ')
    outfile.write(str(best_run))
    global_config.best_model.save('AutoencoderN.h5')
    outfile.close()
