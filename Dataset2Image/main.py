import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle

from keras import Model
from keras.engine.saving import load_model


import pandas as pd
import csv
from Dataset2Image.lib import DeepInsight_train_norm
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Parameters
param = {"Max_A_Size": 10, "Max_B_Size": 10, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180,
         "dir": "dataset/CICDS2017/", "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
         "LoadFromJson": False, "mutual_info": False, # Mean or MI
         "hyper_opt_evals": 50, "epoch": 40, "No_0_MI": False,  # True -> Removing 0 MI Features
         "autoencoder": False
         }

# TODO delete
# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]

if not param["LoadFromJson"]:
    data = {}
    with open(param["dir"] + 'Train.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"]["classification"]
        del data["Xtrain"]["classification"]
    with open(param["dir"] + 'Test_UNSW_NB15.csv', 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest.astype(float)
        data["Ytest"] = data["Xtest"]["classification"]
        del data["Xtest"]["classification"]
    # data["Ytest"] = 1
    # data["Xtest"] = 1
    # MI=mutual_info_classif(data["Xtrain"],data["Classification"])

    if param["No_0_MI"]:
        with open(param["dir"] + '0_MI.json') as json_file:
            j = json.load(json_file)
        data["Xtrain"] = data["Xtrain"].drop(columns=j)
        data["Xtest"] = data["Xtest"].drop(columns=j)
        print("0 MI features dropped!")

     # AUTOENCODER
    if param["autoencoder"]:
        autoencoder = load_model(param["dir"] + 'Autoencoder.h5')
        autoencoder.summary()
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encod2').output)
        encoder.summary()
        # usa l'encoder con predict sul train_X e poi su test_X. Io qui ho creato anche il dataframe per salvarlo poi come csv
        encoded_train = pd.DataFrame(encoder.predict(data["Xtrain"]))
        data["Xtrain"] = encoded_train.add_prefix('feature_')
        encoded_test = pd.DataFrame(encoder.predict(data["Xtest"]))
        data["Xtest"] = encoded_test.add_prefix('feature_')

        f_myfile = open(param["dir"] + 'Xtrain_auto.pickle', 'wb')
        pickle.dump(data["Xtrain"], f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'Xtest_auto.pickle', 'wb')
        pickle.dump(data["Xtest"], f_myfile)
        f_myfile.close()


    # f_myfile = open(param["dir"] + 'YTrain.pickle', 'wb')
    # pickle.dump(data["Classification"], f_myfile)
    # f_myfile.close()
    #
    # f_myfile = open(param["dir"] + 'YTest.pickle', 'wb')
    # pickle.dump(data["Ytest"], f_myfile)
    # f_myfile.close()

    model = DeepInsight_train_norm.train_norm(param, data, norm=False)

else:
    images = {}
    f_myfile = open(param["dir"] + 'train_8x8_Mean.pickle', 'rb')
    images["Xtrain"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'YTrain.pickle', 'rb')
    images["Classification"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'test_8x8_Mean.pickle', 'rb')
    images["Xtest"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open(param["dir"] + 'YTest.pickle', 'rb')
    images["Ytest"] = pickle.load(f_myfile)
    f_myfile.close()

    # with open('dataset/CICDS2017/Test.csv', 'r') as file:
    #     Xtest=pd.DataFrame(list(csv.DictReader(file)))
    #     Xtest.replace("", np.nan, inplace=True)
    #     Xtest.dropna(inplace=True)
    #     images["Ytest"] = Xtest["Classification"]

    model = DeepInsight_train_norm.train_norm(param, images, norm=False)
