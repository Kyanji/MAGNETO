import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pandas as pd
import csv
from Dataset2Image.lib import DeepInsight_train_norm
import cv2


# Parameters
param = {"Max_P_Size": 33, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180, "Mode": "neural",
         "LoadFromJson": True}

# TODO delete
# with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]

if not param["LoadFromJson"]:
    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        data["Classification"] = data["Xtrain"]["Classification"]
        del data["Xtrain"]["Classification"]

    model=DeepInsight_train_norm.train_norm(param, data, norm=True)
    model.save('dataset/CICDS2017/param/model.h5')

else:

    filenames = glob.glob("dataset/CICDS2017/images/*.jpg")
    filenames.sort()
    images = [cv2.imread(img, 0) for img in filenames]

    with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        images = {"Xtrain": images, "Classification": data["Xtrain"]["Classification"]}
    model=DeepInsight_train_norm.train_norm(param, images, norm=True)
    model.save('dataset/CICDS2017/param/model.h5')

    with open('dataset/CICDS2017/Test.csv', 'r') as file:
        test = {"Xtest": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
        test["Classification"] = test["Xtest"]["Classification"]
        del test["Xtrain"]["Classification"]



