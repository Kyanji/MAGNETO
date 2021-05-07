# iMAge based Gan enhanced convolutional NEural neTwOrk  (MAGNETO)


The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Luca De Rose, Donato Malerba_

[GAN Augmentation to Deal with Imbalance in Imaging-based  Intrusion Detection](https://www.sciencedirect.com/science/article/pii/S0167739X21001382) 

Please cite our work if you find it useful for your research and work.
```
 @article{ANDRESINI2021108,
title = {GAN augmentation to deal with imbalance in imaging-based intrusion detection},
journal = {Future Generation Computer Systems},
volume = {123},
pages = {108-127},
year = {2021},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2021.04.017},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X21001382},
author = {Giuseppina Andresini and Annalisa Appice and Luca {De Rose} and Donato Malerba}}
```

![MAGNETO](https://ars.els-cdn.com/content/image/1-s2.0-S0167739X21001382-gr1_lrg.jpg)

## Code requirements
The code relies on the following python3.7+ libs.
Packages needed are:
* Tensorflow 2.1.0
* Pandas 1.0.1
* Numpy 1.18.1
* Imbalanced-learn 0.7
* Hyperopt 0.2
* Keras 2.3.1
* Sklearn
* Pickle

## Data
The following [DATASETS](https://drive.google.com/drive/folders/1lzl_O29ewRwfiRjV3g4m69ot0pLQYJy3?usp=sharing) are trasformed to have a binary classification task (attacks or normal traffic).
The datasets used are:
* CICIDS2017
* UNSW-NB15
* AAGM17
* KDD-CUP99

## How to use

The repository contains the following scripts:
* main.py:  script to execute MAGNETO 
* train.py : script to execute the whole learning and testing task
* Cart2Pixel : script to create a mapping between examples into images
* ConvPixel : script that use a role to convert an array into images
* deep.py :  script that contain two neural networks
* MinRect.py : script that calculate the minimum rectangle containing all the points in an array
* AGAN.py : script that create the ACGAN
* gan.py : script that train the ACGAN
* Generator.py : script to create examples from an ACGAN

## Replicate the experiments
Modify the following code in the main.py script to change the beaviour of MAGNETO

# Parameters
```python
param = {"Max_A_Size": 10,  # Heigth and Weight of the images
         "Max_B_Size": 10, 
         "Dynamic_Size": False,  # search the minimum A and B to create 0 Collisions
         'Metod': 'tSNE',   # {tSNE, kpca, pca} to create the mapping between examples and images 
         "ValidRatio": 0.1, 
         "seed": 180,
         "dir": "dataset/dataset4/",  # path of dataset
         "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
         "LoadFromPickle": False, # load dataset images from pickle
         "mutual_info": False,  # Mean or MI
         "hyper_opt_evals": 50, 
         "epoch": 200,
         "No_0_MI": False,  # True : remove 0 MI Features
         "autoencoder": False, # use autoencoder to reduce the number of features
         "enhanced_dataset": "gan"  # gan, smote, adasyn, ""None""
         }
```









