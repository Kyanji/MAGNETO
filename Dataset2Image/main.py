import os
import pandas as pd
import csv
from Dataset2Image.lib import DeepInsight_train_norm
from os import listdir
from os.path import isfile, join
import cv2


# Parameters
param = {"Max_P_Size": 33, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180, "Mode": "neural",
         "LoadFromJson": True}

# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]
if not param["LoadFromJson"]:
    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
    DeepInsight_train_norm.train_norm(param, data, norm=1)
else:

    mypath = 'dataset/CICDS2017/images/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = []
    for n in range(0, len(onlyfiles)):
        images.append(cv2.imread(join(mypath, onlyfiles[n]), 0))
    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
    images = {"Xtrain": images, "Classification": data["Xtrain"]["Classification"]}
    DeepInsight_train_norm.train_norm(param, images, norm=1)
