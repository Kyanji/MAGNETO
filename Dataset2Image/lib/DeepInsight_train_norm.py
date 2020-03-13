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
from Dataset2Image.lib.deep import deep_train

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []


def hyperopt_fcn(params):
    global SavedParameters
    start = timeit.timeit()
    # for p in SavedParameters:
    #     if p["filter"] == params["filter"] and p["filter2"] == params["filter2"] and p["kernel"] == params["kernel"] and\
    #             p["learning_rate"] == params["learning_rate"] and p["momentum"] == params["momentum"]:
    #         return {'loss': np.inf, 'status': STATUS_OK}

    model = deep_train(XGlobal, YGlobal, params)
    Y_predicted = model.predict(XTestGlobal, verbose=0, use_multiprocessing=True, workers=12)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    end = timeit.timeit()

    cf = confusion_matrix(YTestGlobal, Y_predicted)
    print(cf)
    print(balanced_accuracy_score(YTestGlobal, Y_predicted))
    K.clear_session()
    # SavedParameters.append(
    #     {"balanced_accuracy": balanced_accuracy_score(YTestGlobal, Y_predicted) * 100, "TN": cf[0][0],
    #      "FP": cf[0][1], "FN": cf[1][0], "TP": cf[1][1], "filter": params["filter"], "filter2": params["filter2"],
    #      "kernel": params["kernel"], "learning_rate": params["learning_rate"], "momentum": params["momentum"]})
    SavedParameters.append(
        {"balanced_accuracy": balanced_accuracy_score(YTestGlobal, Y_predicted) * 100, "TN": cf[0][0],
         "FP": cf[0][1], "FN": cf[1][0], "TP": cf[1][1], "kernel": params["kernel"],
         "learning_rate": params["learning_rate"],
         "batch": params["batch"], "time": end - start
         })

    try:
        with open("param.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")

    return {'loss': -balanced_accuracy_score(YTestGlobal, Y_predicted), 'status': STATUS_OK}


def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])

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

        q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
             "max_px_size": param["Max_P_Size"]}
        print(q["method"])
        print(q["max_px_size"])

        # generate images
        XGlobal, image_model = Cart2Pixel(q, q["max_px_size"], q["max_px_size"], param["Dynamic_Size"])

        # generate testingset image
        dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()

        XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                 image_model["A"], image_model["B"]) for i in range(0, dataset["Xtest"].shape[1])]

        del dataset["Xtest"]

        # saving testingset
        filename = "dataset/CICDS2017/param/testingsetImage.pickle"
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
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size, image_size, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    # optimizable_variable = {"filter_size": 3, "kernel": 2, "filter_size2": 6,"learning_rate":1e-5,"momentum":0.8}
    optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 16 + 1)),
                            "batch": hp.choice("batch", [64, 128, 256, 512]),
                            "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01)}

    trials = Trials()
    global SavedParameters
    SavedParameters150 = []
    # with open('param.csv') as csv_file:
    #     SavedParameters150 = list(csv.DictReader(csv_file))
    #
    # for i in range(8):
    #     param = {"kernel": int(SavedParameters150[i]["kernel"]),
    #              "filter": int(SavedParameters150[i]["filter"]),
    #              "filter2": int(SavedParameters150[i]["filter2"]),
    #              "learning_rate": float(SavedParameters150[i]["learning_rate"]),
    #              "momentum": float(SavedParameters150[i]["momentum"])
    #              }
    #     hyperopt_fcn(param)
    fmin(hyperopt_fcn, optimizable_variable, trials=trials, algo=tpe.suggest, max_evals=20)

    # print("migliori parametri")
    # SavedParameters = sorted(SavedParameters, key=lambda i: i['balanced_accuracy'], reverse=True)
    # print(SavedParameters[0])

    # returning best model with hyperopt parameters
    model = deep_train(XGlobal, YGlobal, SavedParameters[0])
    print("accuracy" + str(model))
    return model
