import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from lib.Cart2Pixel import Cart2Pixel

def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    dataset["Xtrain"], dataset["XValidation"], y_train, y_val = train_test_split(dataset["Xtrain"],
                                                                                 dataset["Xtrain"]["Classification"],
                                                                                 test_size=param["ValidRatio"],
                                                                                 random_state=param["seed"])
    del dataset["Xtrain"]["Classification"]
    # dataset["Xtrain"]=dataset["Xtrain"].T
    # norm
    Out = {}
    if norm == 1:
        Out["Norm"] = 1
        print('NORM-1')
        Out["Max"] = float(dataset["Xtrain"].max().max())
        Out["Min"] = float(dataset["Xtrain"].min().min())
        #NORM
        dataset["Xtrain"]=(dataset["Xtrain"]- Out["Min"])/( Out["Max"]- Out["Min"])
        dataset["XValidation"]=(dataset["XValidation"]- Out["Min"])/( Out["Max"]- Out["Min"])

        dataset["Xtrain"]=dataset["Xtrain"].fillna(0)
        dataset["XValidation"]=dataset["XValidation"].fillna(0)

    #Flip

    # TODO implement norm 2
    q = {}
    q["data"] = np.array(dataset["Xtrain"].values).transpose()
    q["method"] = 'tSNE'
    q["max_px_size"] = 120
    Cart2Pixel(q, q["max_px_size"], q["max_px_size"])

    print("done")
