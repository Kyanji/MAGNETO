import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from lib.Cart2Pixel import Cart2Pixel


def train_norm(param, dataset,norm):
    np.random.seed(param["seed"])
    true_label = np.array([])
    for j in range(1, dataset["class"] + 1):
        true_label = np.append(true_label, np.ones((1, dataset["num_tr"][j - 1])) * j)
    #true_label = np.reshape(true_label, (len(true_label), 1))

    q = range(1, len(true_label) + 1)
    y_train=true_label[:, np.newaxis].tolist() #vertical list
    idx = []
    for j in range(1, dataset["class"] + 1):
        rng = []
        idx.append([])
        for i in range(len(true_label)):
            if true_label.item(i) == (j):  # class j+1
                rng.append(q[i])
        idx[j - 1]=np.random.randint(len(rng), size=int(round(len(rng) * param["ValidRatio"],0)),dtype="int")
    idx = [item for sublist in idx for item in sublist] # flat list
    #split dataset into val and training
    dataset["Xtrain"] = np.array(dataset["Xtrain"])
    dataset["XValidation"]=[]
    for i in idx:
        dataset["XValidation"].append(dataset["Xtrain"][:, i])
        dataset["Xtrain"]=np.delete(dataset["Xtrain"],i,1)
    dataset["XValidation"]=np.array(dataset["XValidation"]).transpose()
    print(dataset["XValidation"].shape)
    y_validation=[y_train[index] for index in idx]
    for i in idx:
        y_train=np.delete(y_train,i,0)

    # norm
    Out={}
    if norm==1:
        Out["Norm"] = 1
        print('NORM-1')
        Out["Max"] = dataset["Xtrain"].max()
        Out["Min"] = dataset["Xtrain"].min()
        for i in range(len(dataset["Xtrain"])):
            for j in range(len(dataset["Xtrain"][i])):
                dataset["Xtrain"][i][j]=(dataset["Xtrain"][i][j] - Out["Min"]) / (Out["Max"] - Out["Min"])
                if dataset["Xtrain"][i][j]>1:
                    dataset["Xtrain"][i][j] = 1
                elif dataset["Xtrain"][i][j]<0:
                    dataset["Xtrain"][i][j] = 0
        for i in range(len(dataset["XValidation"])):
            for j in range(len(dataset["XValidation"][i])):
                dataset["XValidation"][i][j]=(dataset["XValidation"][i][j] - Out["Min"]) / (Out["Max"] - Out["Min"])
                if dataset["XValidation"][i][j]>1:
                    dataset["XValidation"][i][j] = 1
                elif dataset["XValidation"][i][j]<0:
                    dataset["XValidation"][i][j] = 0
        where_are_NaNs = np.isnan( dataset["Xtrain"])
        where_are_NaNs_val = np.isnan( dataset["XValidation"])
        dataset["Xtrain"][where_are_NaNs]=0
        dataset["XValidation"][where_are_NaNs_val]=0
    # TODO implement norm 2
    q={}
    q["data"]=np.array(dataset["Xtrain"])
    q["method"]='tSNE'
    q["max_px_size"]=120
    Cart2Pixel(q,q["max_px_size"],q["max_px_size"])



    print("done")
