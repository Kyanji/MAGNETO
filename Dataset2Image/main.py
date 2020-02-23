import json
from lib.DeepInsight_train import train

# Parameters
param = {"Max_P_Size": 120, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180}

with open('dataset/exptable.txt') as json_file:
    data = json.load(json_file)["dset"]
train(param, data, norm=1)
