from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)

pathModel='acgan_aagm.h5'
generator = load_model(pathModel)
param = {"Max_A_Size": 10, "Max_B_Size": 10, "Dynamic_Size": False, 'Metod': 'tSNE', "ValidRatio": 0.1, "seed": 180,
             "dir": "dataset/AAGM/", "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
             "LoadFromJson": False, "mutual_info": True,  # Mean or MI
             "hyper_opt_evals": 20, "epoch": 150, "No_0_MI": False,  # True -> Removing 0 MI Features
             "autoencoder": False, "cut": None
             }
dim=26665
images = {}
if param['mutual_info']:
    method = 'MI'
else:
    method = 'Mean'
f_myfile = open(param["dir"] +'ok/XTrain40A%.pickle','rb')
images["Xtrain"] = pickle.load(f_myfile)
f_myfile.close()

f_myfile = open(param["dir"] + 'ok/YTrain40A%.pickle', 'rb')
images["Classification"] = pickle.load(f_myfile)
(x_train, y_train) = np.asarray(images["Xtrain"]), np.asarray(images["Classification"])
print(x_train.shape)


print(y_train.shape)

noise_input = np.random.uniform(-1.0, 1.0, size=[dim, 100]) #se 1 produce 1 sola immagine
#AGAN#

class_label=0
noise_label = np.zeros((dim, 2))
noise_label[:,class_label] = 1
step = class_label
noise_input = [noise_input, noise_label]

#AGAN
generator.summary()
predictions = generator.predict(noise_input)
predictions=tf.reshape(predictions,[dim,10,10])

print(1, type(images["Xtrain"]))
print(2, type(predictions.numpy().tolist()))
new = predictions.numpy().tolist()
print(len(new))
images["Xtrain"].extend(new)
print(4, type(images["Classification"].tolist()))
print(4, type(list(np.zeros(dim))))

images["Classification"] = images["Classification"].append(pd.Series(np.zeros(dim)))
print(len(images['Xtrain']))

f_myfile = open(param["dir"] + 'XTrain50A%.pickle', 'wb')
pickle.dump(images["Xtrain"], f_myfile)
f_myfile.close()

f_myfile = open(param["dir"] + 'YTrain50A%.pickle', 'wb')
pickle.dump(images["Classification"], f_myfile)
f_myfile.close()
