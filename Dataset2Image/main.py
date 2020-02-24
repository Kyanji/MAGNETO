import json
from lib.DeepInsight_train_norm import train_norm
import pandas as pd
import csv
import numpy as np
# Parameters
param = {"Max_P_Size": 120,"Dynamic_Size":True , 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180}

#with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]
with open('dataset/CICDS2017/TrainOneCls.csv', 'r') as file:
    data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
train_norm(param, data, norm=1)
