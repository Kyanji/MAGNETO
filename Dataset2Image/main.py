import os
import pickle
import pandas as pd
import csv
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from Dataset2Image.lib import train
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Parameters
param = {"Max_A_Size": 10, "Max_B_Size": 10, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180,
         "dir": "dataset/dataset4/", "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
         "LoadFromPickle": False, "mutual_info": False,  # Mean or MI
         "hyper_opt_evals": 50, "epoch": 2, "No_0_MI": False,  # True -> Removing 0 MI Features
         "autoencoder": False, "cut": None, "enhanced_dataset": "gan"  # gan, smote, adasyn, ""None""
         }

dataset = 4  # change dataset
if dataset == 1:
    train = 'TrainOneCls.csv'
    test = 'Test.csv'
    classif_label = 'Classification'
    param["attack_label"] = 0
elif dataset == 2:
    train = 'Train.csv'
    test = 'Test_UNSW_NB15.csv'
    classif_label = 'classification'
    param["attack_label"] = 1
elif dataset == 3:
    train = 'Train.csv'
    test = 'Test.csv'
    classif_label = ' classification.'
    param["attack_label"] = 1
elif dataset == 4:
    train = 'AAGMTrain_OneClsNumeric.csv'
    test = 'AAGMTest_OneClsNumeric.csv'
    classif_label = 'classification'
    param["attack_label"] = 0

if not param["LoadFromPickle"]:
    data = {}
    with open(param["dir"]+train, 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"][classif_label]
        del data["Xtrain"][classif_label]

    with open(param["dir"] + test, 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        print(Xtest.shape)
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest.astype(float)
        data["Ytest"] = data["Xtest"][classif_label]
        del data["Xtest"][classif_label]

    if param["enhanced_dataset"] == "smote":
        sm = SMOTE(random_state=42)
        data["Xtrain"], data["Classification"] = sm.fit_resample(data["Xtrain"], data["Classification"])
    elif param["enhanced_dataset"] == "adasyn":
        ada = ADASYN(random_state=42)
        data["Xtrain"], data["Classification"] = ada.fit_resample(data["Xtrain"], data["Classification"])

    model = train.train_norm(param, data, norm=False)

else:
    images = {}
    if param['mutual_info']:
        method = 'MI'
    else:
        method = 'Mean'

    if param["enhanced_dataset"] == "gan":
        f_myfile = open(param["dir"] + 'XTrain50A%.pickle', 'rb')
        images["Xtrain"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'YTrain50A%.pickle', 'rb')
        images["Classification"] = pickle.load(f_myfile)
        f_myfile.close()
    else:
        f_myfile = open(param["dir"] + 'train_' + str(param['Max_A_Size']) + 'x' + str(
            param['Max_B_Size']) + '_' + method + '.pickle', 'rb')
        images["Xtrain"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'YTrain.pickle', 'rb')
        images["Classification"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(
            param["dir"] + 'test_' + str(param['Max_A_Size']) + 'x' + str(
                param['Max_B_Size']) + '_' + method + '.pickle',
            'rb')
        images["Xtest"] = pickle.load(f_myfile)
        f_myfile.close()

        f_myfile = open(param["dir"] + 'YTest.pickle', 'rb')
        images["Ytest"] = pickle.load(f_myfile)
        f_myfile.close()

    model = train.train_norm(param, images, norm=False)

# PLOT DATA -- PCA/TSNE
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# pca = TSNE(n_components=2)
# principalComponents = pca.fit_transform(data["Xtrain"])
#
#
# scaler = MinMaxScaler(feature_range=(0, 10))
# scaled_data = scaler.fit_transform(principalComponents)
#
# attacks = np.where(data["Classification"] == 0)
# attacks = scaled_data[attacks[0]]
#
# normals = np.where(data["Classification"] == 1)
# normals = scaled_data[normals[0]]
#
# plt.scatter(attacks[:, 0], attacks[:, 1], color="red", s=1)
#
# plt.scatter(normals[:, 0], normals[:, 1], color="blue", s=1)
#
# plt.show()
# df = pd.DataFrame(scaled_data)
# df["label"] = data["Classification"]
# df.to_csv("AAGM-TSNE_minmax.csv", index=False)
