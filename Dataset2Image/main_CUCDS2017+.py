import json
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import csv
# from Dataset2Image.lib import DeepInsight_train_norm
import numpy as np

# Parameters
param = {"Max_A_Size": 10, "Max_B_Size": 10, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180,
         "dir": "dataset/CICDS2017/", "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
         "LoadFromJson": False, "mutual_info": True,  # Mean or MI
         "hyper_opt_evals": 50, "epoch": 20, "No_0_MI": False,  # True -> Removing 0 MI Features
         "autoencoder": False, "cut": None
         }

# TODO delete
# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]

if not param["LoadFromJson"]:
    data = {}
    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"]["Classification"]
        del data["Xtrain"]["Classification"]

    # with open('dataset/CICDS2017/Testing.csv', 'r') as file:
    #     Xtest = pd.DataFrame(list(csv.DictReader(file)))
    #     Xtest.replace("", np.nan, inplace=True)
    #     Xtest.dropna(inplace=True)
    #     data["Xtest"] = Xtest.astype(float)
    #
    #     data["Ytest"] = data["Xtest"]["Classification"]
    #     del data["Xtest"]["Classification"]

    # filename = "dataset/CICDS2017/ytrain.pickle"
    # f_myfile = open(filename, 'wb')
    # pickle.dump(data["Ytest"], f_myfile)
    # f_myfile.close()

    from imblearn.over_sampling import SMOTE, ADASYN

    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(data["Xtrain"], data["Classification"])
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data["Xtrain"])
    import matplotlib.pyplot as plt

    #
    # ada = ADASYN(random_state=42)
    # X_res, y_res = ada.fit_resample(data["Xtrain"], data["Classification"])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 10))
    scaled_data = scaler.fit_transform(principalComponents)

    attacks = np.where(data["Classification"] == 0)
    attacks = scaled_data[attacks[0]]

    normals = np.where(data["Classification"] == 1)
    normals = scaled_data[normals[0]]

    plt.scatter(normals[:, 0], normals[:, 1], color="blue",s=1)
    plt.scatter(attacks[:, 0], attacks[:, 1], color="red",s=1)

    plt.show()
    df = pd.DataFrame(scaled_data)
    df["label"] = data["Classification"]
    df.to_csv("CICDS-PCA_minmax.csv", index=False)

    # TODO remove this line
    data["Ytest"] = 1
    data["Xtest"] = data["Xtrain"]

    # model = DeepInsight_train_norm.train_norm(param, data, norm=False)
    # model.save('dataset/CICDS2017/param/model.h5')

else:
    images = {}
    f_myfile = open('dataset/CICDS2017/train_10x10_MI.pickle', 'rb')
    images["Xtrain"] = pickle.load(f_myfile)
    f_myfile.close()

    f_myfile = open('dataset/CICDS2017/ytrain.pickle', 'rb')
    images["Classification"] = pickle.load(f_myfile)
    f_myfile.close()

    # f_myfile = open('dataset/CICDS2017/test.pickle', 'rb')
    # images["Xtest"] = pickle.load(f_myfile)
    # f_myfile.close()
    #
    # f_myfile = open('dataset/CICDS2017/y_testingset.pickle', 'rb')
    # images["Ytest"] = pickle.load(f_myfile)
    # f_myfile.close()

    f_myfile = open('dataset/CICDS2017/gan_images10x10MI.pickle', 'rb')
    gan = pickle.load(f_myfile)
    f_myfile.close()

    # with open('dataset/CICDS2017/Test.csv', 'r') as file:
    #     Xtest=pd.DataFrame(list(csv.DictReader(file)))
    #     Xtest.replace("", np.nan, inplace=True)
    #     Xtest.dropna(inplace=True)
    #     images["Ytest"] = Xtest["Classification"]
    x = np.concatenate((images["Xtrain"], gan))
    y = np.concatenate((images["Classification"], np.zeros(4000)))
    # model = DeepInsight_train_norm.train_norm(param, images, norm=False)
