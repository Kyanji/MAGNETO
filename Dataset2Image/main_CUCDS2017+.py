import json
import os
import pickle

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import csv
from Dataset2Image.lib import DeepInsight_train_norm
import numpy as np

# Parameters
param = {"Max_P_Size": 10, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180, "Mode": "neural",
         "LoadFromJson": True}

# TODO delete
# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]

if not param["LoadFromJson"]:
    data = {}
    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"]["Classification"]
        del data["Xtrain"]["Classification"]

    # with open('dataset/CICDS2017/Test.csv', 'r') as file:
    #     Xtest = pd.DataFrame(list(csv.DictReader(file)))
    #     Xtest.replace("", np.nan, inplace=True)
    #     Xtest.dropna(inplace=True)
    #     data["Xtest"] = Xtest.astype(float)
    #
    #     data["Ytest"] = data["Xtest"]["Classification"]
    #     del data["Xtest"]["Classification"]

    data["Xtest"]=data["Xtrain"]
    data["Ytest"]=data["Classification"]


    #
    # filename = "dataset/CICDS2017/param/y_testingset.pickle"
    # f_myfile = open(filename, 'wb')
    # pickle.dump(data["Ytest"], f_myfile)
    # f_myfile.close()
    # TODO remove this line
    # data["Ytest"] = 1
    # data["Xtest"] = data["Xtrain"]

    model = DeepInsight_train_norm.train_norm(param, data, norm=False)
    model.save('dataset/CICDS2017/param/model.h5')

else:
    images = {}
    f_myfile = open('dataset/CICDS2017/param/trainingsetImageNew10x10.pickle', 'rb')
    images["Xtrain"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open('dataset/CICDS2017/param/y_trainingset.pickle', 'rb')
    images["Classification"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open('dataset/CICDS2017/param/testingsetImageNew10x10.pickle', 'rb')
    images["Xtest"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open('dataset/CICDS2017/param/y_testingset.pickle', 'rb')
    images["Ytest"] = pickle.load(f_myfile)
    f_myfile.close()

    # with open('dataset/CICDS2017/Test.csv', 'r') as file:
    #     Xtest=pd.DataFrame(list(csv.DictReader(file)))
    #     Xtest.replace("", np.nan, inplace=True)
    #     Xtest.dropna(inplace=True)
    #     images["Ytest"] = Xtest["Classification"]

    model = DeepInsight_train_norm.train_norm(param, images, norm=False)
