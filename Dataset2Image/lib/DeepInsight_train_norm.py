import csv
import json
import pickle
import timeit

import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

from Dataset2Image.lib.Cart2Pixel import Cart2Pixel
from Dataset2Image.lib.ConvPixel import ConvPixel
from Dataset2Image.lib.deep import CNN_Nature, CNN2
import matplotlib.pyplot as plt

import time

XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
Name = ""
best_val_acc = 0

attack_label = 0


def fix(f):
    a = f["TN_val"]
    b = f["FP_val"]
    c = f["FN_val"]
    d = f["TP_val"]
    f["TN_val"] = d
    f["TP_val"] = a
    f["FP_val"] = c
    f["FN_val"] = b
    return f


def fix_test(f):
    a = f["TN_test"]
    b = f["FP_test"]
    c = f["FN_test"]
    d = f["TP_test"]
    f["TN_test"] = d
    f["TP_test"] = a
    f["FP_test"] = c
    f["FN_test"] = b
    return f


def res(cm, val):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    print(attacks)
    print(normals)

    if attacks <= normals:
        print("ok")
    elif not val:
        print("error")
        return False, False
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return True, r


# hyperopt function to optimize
def hyperopt_fcn(params):
    if Mode == "CNN_Nature" and params["filter"] == params["filter2"]:
        return {'loss': np.inf, 'status': STATUS_OK}
    global SavedParameters
    start_time = time.time()
    print("start train")
    if Mode == "CNN_Nature":
        model, val = CNN_Nature(XGlobal, YGlobal, params)
    elif Mode == "CNN2":
        model, val = CNN2(XGlobal, YGlobal, params)
    print("start predict")

    y_predicted = model.predict(XTestGlobal, verbose=0, use_multiprocessing=True, workers=12)
    y_predicted = np.argmax(y_predicted, axis=1)
    elapsed_time = time.time() - start_time
    cf = confusion_matrix(YTestGlobal, y_predicted)
    # print(cf)
    # print("test F1_score: " + str(f1_score(YTestGlobal, y_predicted)))
    K.clear_session()
    SavedParameters.append(val)
    global best_val_acc
    # print("val acc: " + str(val["F1_score_val"]))

    if Mode == "CNN_Nature":
        SavedParameters[-1].update({"balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, y_predicted) *
                                                              100, "TP_test": cf[0][0], "FN_test": cf[0][1],
                                    "FP_test": cf[1][0], "TN_test": cf[1][1], "kernel": params[
                "kernel"], "learning_rate": params["learning_rate"], "batch": params["batch"],
                                    "filter1": params["filter"],
                                    "filter2": params["filter2"],
                                    "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})
    elif Mode == "CNN2":
        SavedParameters[-1].update(
            {"balanced_accuracy_test": balanced_accuracy_score(YTestGlobal, y_predicted) * 100, "TP_test": cf[0][0],
             "FN_test": cf[0][1], "FP_test": cf[1][0], "TN_test": cf[1][1], "kernel": params["kernel"],
             "learning_rate": params["learning_rate"],
             "batch": params["batch"],
             "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})
    if attack_label == 0:
        SavedParameters[-1] = fix(SavedParameters[-1])
        cm_val = [[SavedParameters[-1]["TP_val"], SavedParameters[-1]["FN_val"]],
                  [SavedParameters[-1]["FP_val"], SavedParameters[-1]["TN_val"]]]

    done, r = res(cm_val, True)
    assert done == True
    SavedParameters[-1].update({
        "OA_val": r[0],
        "P_val": r[2],
        "R_val": r[3],
        "F1_val": r[4],
        "FAR_val": r[5],
        "TPR_val": r[6]
    })
    cm_test = [[SavedParameters[-1]["TP_test"], SavedParameters[-1]["FN_test"]],
               [SavedParameters[-1]["FP_test"], SavedParameters[-1]["TN_test"]]]
    if attack_label == 0:
        SavedParameters[-1] = fix_test(SavedParameters[-1])
        cm_test = [[SavedParameters[-1]["TP_test"], SavedParameters[-1]["FN_test"]],
                   [SavedParameters[-1]["FP_test"], SavedParameters[-1]["TN_test"]]]
    done, r = res(cm_test, False)
    assert done == True
    # Save model
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        model.save(Name.replace(".csv", "_model.h5"))
        best_val_acc = SavedParameters[-1]["F1_val"]

    SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

    try:
        with open(Name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': -val["F1_val"], 'status': STATUS_OK}


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

        print("trasposing")

        q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
             "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": np.argmax(YGlobal, axis=1)}
        print(q["method"])
        print(q["max_A_size"])
        print(q["max_B_size"])

        # generate images
        XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                    mutual_info=param["mutual_info"], params=param, only_model=False)

        del q["data"]
        print("Train Images done!")
        # generate testing set image
        if param["mutual_info"]:
            dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

        x = image_model["xp"]
        y = image_model["yp"]
        col = dataset["Xtest"].columns
        # col = col.delete(0)
        # print(col)
        # coordinate model
        coor_model = {"coord": ["xp: " + str(i) + "," "yp :" + str(z) + ":" + col for i, z, col in zip(x, y, col)]}
        j = json.dumps(coor_model)
        f = open(param["dir"] + "MI_model.json", "w")
        f.write(j)
        f.close()

        dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()
        print("generating Test Images")
        print(dataset["Xtest"].shape)

        if image_model["custom_cut"] is not None:
            XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                     image_model["A"], image_model["B"], custom_cut=range(0, image_model["custom_cut"]))
                           for i in range(0, dataset["Xtest"].shape[1])]  # dataset["Xtest"].shape[1])]
        else:
            XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                     image_model["A"], image_model["B"])
                           for i in range(0, dataset["Xtest"].shape[1])]  # dataset["Xtest"].shape[1])]

        print("Test Images done!")

        # saving testing set
        name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))
        if param["No_0_MI"]:
            name = name + "_No_0_MI"
        if param["mutual_info"]:
            name = name + "_MI"
        else:
            name = name + "_Mean"
        if image_model["custom_cut"] is not None:
            name = name + "_Cut" + str(image_model["custom_cut"])
        filename = param["dir"] + "test" + name + ".pickle"
        f_file = open(filename, 'wb')
        pickle.dump(XTestGlobal, f_file)
        f_file.close()
    else:
        XGlobal = dataset["Xtrain"]
        XTestGlobal = dataset["Xtest"]
    # GAN
    del dataset["Xtrain"]
    del dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size1, image_size2 = XTestGlobal[0].shape
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size1, image_size2, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    # hyperparameters_to_optimize = {"filter_size": 3, "kernel": 2, "filter_size2": 6,"learning_rate":1e-5,
    # "momentum":0.8}

    if param["Mode"] == "CNN_Nature":
        hyperparameters_to_optimize = {"kernel": hp.choice("kernel", np.arange(2, 4 + 1)),
                                       "filter": hp.choice("filter", [16, 32, 64, 128]),
                                       "filter2": hp.choice("filter2", [16, 32, 64, 128]),
                                       "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                       "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01),
                                       "epoch": param["epoch"]}
    elif param["Mode"] == "CNN2":
        hyperparameters_to_optimize = {"kernel": hp.choice("kernel", np.arange(2, 4 + 1)),
                                       "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                       'dropout1': hp.uniform("dropout1", 0, 1),
                                       'dropout2': hp.uniform("dropout2", 0, 1),
                                       "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
                                       "epoch": param["epoch"]}

    # output name
    global attack_label
    attack_label = param["attack_label"]

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
    fmin(hyperopt_fcn, hyperparameters_to_optimize, trials=trials, algo=tpe.suggest, max_evals=param["hyper_opt_evals"])

    print("done")
    return 1
