import numpy as np
from bayes_opt import BayesianOptimization
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K

from Dataset2Image.lib.Cart2Pixel import Cart2Pixel
from Dataset2Image.lib.deep_base_model import model_train_with_val, model_train
from hyperopt import STATUS_OK

XGlobal = []
YGlobal = []


def hyperopt_fcn(params):
    model = model_train_with_val(XGlobal, YGlobal, params)
    K.clear_session()
    return {'loss': -model, 'status': STATUS_OK}


def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    y = dataset["Classification"]
    del dataset["Classification"]

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
        dataset["Xtrain"] = Cart2Pixel(q, q["max_px_size"], q["max_px_size"], param["Dynamic_Size"])

    # optimizable_variable = {"filter_size": 3, "kernel": 2, "filter_size2": 6,"learning_rate":1e-5,"momentum":0.8}
    optimizable_variable = {"filter_size": hp.choice("filter_size", np.arange(2, 10 + 1)),
                            "kernel": hp.choice("kernel", np.arange(2, 16 + 1)),
                            "filter_size2": hp.choice("filter_size2", np.arange(4, 30 + 1)),
                            "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-1),
                            "momentum": hp.uniform("momentum", 0.8, 0.95)}

    global XGlobal
    XGlobal = dataset["Xtrain"]
    global YGlobal
    YGlobal = y
    trials = Trials()
    best = fmin(hyperopt_fcn, optimizable_variable, algo=tpe.suggest, max_evals=20, trials=trials)
    print("migliori parametri")
    print(best)
    # returning best model with hyperopt parameters
    model = model_train(XGlobal, YGlobal, best)
    return model

    print("done")
