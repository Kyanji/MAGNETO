import json
from lib.DeepInsight_train import train
import pandas as pd
import csv
import numpy as np
# Parameters
param = {"Max_P_Size": 120, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180}

#with open('dataset/exptable.txt') as json_file:
#    data = json.load(json_file)["dset"]
with open('dataset\CICDS2017\TrainOneCls.csv', 'r') as file:
    data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))).astype(float), "class": 2}
train(param, data, norm=1)
