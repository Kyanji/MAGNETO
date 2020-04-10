import csv
import json
import pickle
import timeit

import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from Dataset2Image.lib.Cart2Pixel import Cart2Pixel
from Dataset2Image.lib.ConvPixel import ConvPixel
from Dataset2Image.lib.deep import CNN_Nature, CNN2

import time

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
Name = ""
best_val_acc = 0


def hyperopt_fcn(params):
    if params["filter"] == params["filter2"]:
        return {'loss': np.inf, 'status': STATUS_OK}
    global SavedParameters
    start_time = time.time()
    print("start train")
    if Mode == "CNN_Nature":
        model, val = CNN_Nature(XGlobal, YGlobal, params)
    elif Mode == "CNN2":
        model, val = CNN2(XGlobal, YGlobal, params)
    print("start predict")

    Y_predicted = model.predict(XTestGlobal, verbose=0, use_multiprocessing=True, workers=12)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    elapsed_time = time.time() - start_time
    cf = confusion_matrix(YTestGlobal, Y_predicted)
    #print(cf)
    print("test accuracy: "+str(balanced_accuracy_score(YTestGlobal, Y_predicted)))
    K.clear_session()
    SavedParameters.append(val)
    global best_val_acc
    print("val acc: "+str(val["balanced_accuracy_val"]))
    if SavedParameters[-1]["balanced_accuracy_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        model.save(Name + "model.h5")
        best_val_acc = SavedParameters[-1]["balanced_accuracy_val"]

    if Mode == "CNN_Nature":
        SavedParameters[-1].update({"balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, Y_predicted) *
                                                              100, "TP_test": cf[0][0], "FN_test": cf[0][1],
                                    "FP_test": cf[1][0], "TN_test": cf[1][1], "kernel": params[
                "kernel"], "learning_rate": params["learning_rate"], "batch": params["batch"],
                                    "filter1": params["filter"],
                                    "filter2": params["filter2"],
                                    "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})
    elif Mode == "CNN2":
        SavedParameters[-1].update(
            {"balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, Y_predicted) * 100, "TP_test": cf[0][0],
             "FN_test": cf[0][1], "FP_test": cf[1][0], "TN_test": cf[1][1], "kernel": params["kernel"],
             "learning_rate": params["learning_rate"],
             "batch": params["batch"],
             "filter1": params["filter"],
             "filter2": params["filter2"],
             "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})
    SavedParameters = sorted(SavedParameters, key=lambda i: i['balanced_accuracy_val'], reverse=True)

    try:
        with open(Name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': -val["balanced_accuracy_val"], 'status': STATUS_OK}


def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    print("modelling dataset")
    global YGlobal
    YGlobal = to_categorical(dataset["Classification"])
    del dataset["Classification"]
    global YTestGlobal
    YTestGlobal = to_categorical(dataset["Ytest"])
    del dataset["Ytest"]

    global XGlobal
    global XTestGlobal

    if not param["LoadFromJson"]:
        # norm
        Out = {}
        if norm:
            print('NORM Min-Max')
            Out["Max"] = float(dataset["Xtrain"].max().max())
            Out["Min"] = float(dataset["Xtrain"].min().min())
            # NORM
            dataset["Xtrain"] = (dataset["Xtrain"] - Out["Min"]) / (Out["Max"] - Out["Min"])
            dataset["Xtrain"] = dataset["Xtrain"].fillna(0)

        # TODO implement norm 2
        print("trasposing")

        q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
             "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": np.argmax(YGlobal, axis=1)}
        print(q["method"])
        print(q["max_A_size"])
        print(q["max_B_size"])

        # generate images
        XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                    mutual_info=param["mutual_info"], params=param)

        del q["data"]
        print("Train Images done!")
        # generate testingset image
        if param["mutual_info"]:
            dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

        dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()
        print("generating Test Images")
        print(dataset["Xtest"].shape)
        XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                 image_model["A"], image_model["B"]) for i in
                       range(0, dataset["Xtest"].shape[1])]  # dataset["Xtest"].shape[1])]
        print("Test Images done!")

        # saving testingset
        name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))
        if param["No_0_MI"]:
            name = name + "_No_0_MI"
        if param["mutual_info"]:
            name = name + "_MI"
        else:
            name = name + "_Mean"
        filename = param["dir"] + "test" + name + ".pickle"
        f_myfile = open(filename, 'wb')
        pickle.dump(XTestGlobal, f_myfile)
        f_myfile.close()
    else:
        XGlobal = dataset["Xtrain"]
        XTestGlobal = dataset["Xtest"]
    del dataset["Xtrain"]
    del dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size = XTestGlobal.shape[1]
    print("shape" + str(XTestGlobal.shape))
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size, image_size, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    # optimizable_variable = {"filter_size": 3, "kernel": 2, "filter_size2": 6,"learning_rate":1e-5,"momentum":0.8}

    if param["Mode"] == "CNN_Nature":
        optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 7 + 1)),
                                "filter": hp.choice("filter", [16, 32, 64, 128]),
                                "filter2": hp.choice("filter2", [16, 32, 64, 128]),
                                "batch": hp.choice("batch", [32]),
                                "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01),
                                "epoch": param["epoch"]}
    elif param["Mode"] == "CNN2":
        optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 7 + 1)),
            "filter": hp.choice("filter", [16, 32, 64, 128]),
            "filter2": hp.choice("filter2", [16, 32, 64, 128]),
            "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
            'dropout1': hp.uniform("dropout1", 0, 1),
            'dropout2': hp.uniform("dropout2", 0, 1),
            "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
            "epoch": param["epoch"]}
    global Mode
    Mode = param["Mode"]

    global Name
    Name = param["dir"] + "res_" + str(int(param["Max_A_Size"])) + "x" + str(int(param["Max_B_Size"]))
    if param["No_0_MI"]:
        Name = Name + "_No_0_MI"
    if param["mutual_info"]:
        Name = Name + "_MI"
    else:
        Name = Name + "_Mean"
    Name = Name + "_" + Mode + ".csv"
    trials = Trials()
    fmin(hyperopt_fcn, optimizable_variable, trials=trials, algo=tpe.suggest, max_evals=param["hyper_opt_evals"])

    print("done")
    return 1
