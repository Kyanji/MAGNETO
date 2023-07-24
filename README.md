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
* hyperopt==0.2.7
* keras==2.9.0
* keras_cv_attention_models==1.3.9
* matplotlib==3.5.2
* numpy==1.21.5
* opencv_contrib_python==4.7.0.68
* opencv_python==4.5.5.64
* pandas==1.4.4
* scikit_image==0.19.2
* scikit_learn==1.0.2
* scipy==1.9.1
* tensorflow==2.9.3
* tensorflow_addons==0.19.0
* vit_keras==0.1.0
* wandb==0.13.10


## Data
The [DATASETS](https://unibari-my.sharepoint.com/:f:/g/personal/l_derose_studenti_uniba_it/EvX4MMTwblRGgdLRTVwvzP0Byrw3ak1jEqGhST81vx-BDA?e=86Ocm5) used are:
* CICMalDroid20
* CICMalMem22
* NSL-KDD
* UNSW-NB15

## How to use

The repository contains the following scripts:
* main.py:  script to execute VINCENT 
* config.ini: configuration file
* 

## Replicate the experiments
Modify the following code in the main.py script to change the beaviour of MAGNETO

# Parameters
```python
[SETTINGS]
UseMagnetoEncoding=False : Convert tabular data to Images or load dataset
Dataset = NSL : MALMEM|MALDROID|NSL|UNSW
TrainVIT=False : Train VIT(Teacher) or load the model if false from the VIT_Teacher_Path
TrainVINCENT=False : Train VINCENT(STUDENT) or load the model if false from the VINCENTPath

[VIT_SETTINGS] : Settings of the VIT (Teacher)
[MAGNETO] : Settings about Magneto encoding (e.g. Image size)
[DISTILLATION] : Settings for the VINCENT Training

[**DATASET**]
tabular_dataset_path=..\..\dataset\malmem\ : path of the tabular dataset
tabular_trainfile=train_split_macro_minmaxdeleted.csv : tabular training file
tabular_testfile=test_split_macro_minmaxdeleted.csv : tabular testing file
classification=Family_int : Classification label for the tabular dataset

trainName=train_8x8_MI.pickle : path of the pickle train images
ytrainName=Ytrain_multi.pickle : path of the pickle train label
testName=test_8x8_MI.pickle  : path of the pickle test images
ytestName=Ytest_multi.pickle : path of the pickle test label

toBinaryMap={"0": 0, "1": 1, "2": 1, "3": 1} : used by Magneto to encode the dataset using binary labels
OutputDirMagneto = MAGNETO_out\malmem\ : Output files for Magneto
OutputDir= .\res\malmem\  :  Output files for VINCENT
VIT_Teacher_Path=./res/malmem/2023-04-06-14-04-51.h5  :  Teacher model
VINCENTPath=./res/malmem/models/PESI.tf :  VINCENT model
Baseline=./res/malmem/cnn2023-05-15-12-54-54/20.tf   :  Baseline (CNN) model


```









