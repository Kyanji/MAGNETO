import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from Dataset2Image.lib.Cart2Pixel import Cart2Pixel
from Dataset2Image.lib.deep_base_model import cnn1


def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    if not param["LoadFromJson"]:
        y_train = dataset["Xtrain"]["Classification"]
        del dataset["Xtrain"]["Classification"]

        # norm
        Out = {}
        if norm == 7:
            Out["Norm"] = 1
            print('NORM Min-Max')
            Out["Max"] = float(dataset["Xtrain"].max().max())
            Out["Min"] = float(dataset["Xtrain"].min().min())
            # NORM
            dataset["Xtrain"] = (dataset["Xtrain"] - Out["Min"]) / (Out["Max"] - Out["Min"])
            # dataset["XValidation"]=(dataset["XValidation"]- Out["Min"])/( Out["Max"]- Out["Min"])

            dataset["Xtrain"] = dataset["Xtrain"].fillna(0)
        # dataset["XValidation"]=dataset["XValidation"].fillna(0)

        # TODO implement norm 2

        if param["Mode"] == "neuralll":

            # import tensorflow as tf
            #
            # print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
            #
            cnn1(dataset["Xtrain"], y_train)

        else:
            q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
                 "max_px_size": param["Max_P_Size"]}
            print(q["method"])
            print(q["max_px_size"])
            images = Cart2Pixel(q, q["max_px_size"], q["max_px_size"], param["Dynamic_Size"])

            cnn1(images, y_train[0:30000])
    else:

        cnn1(dataset["Xtrain"],dataset["Classification"] )
    print("done")
