import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lib.MinRect import minimum_bounding_rectangle
import matplotlib.pyplot as plt

import cv2
import numpy as np


def Cart2Pixel(Q=None, A=None, B=None, *args, **kwargs):
    # TODO controls on input

    # to dataframe
    feat_cols = ["col-" + str(i + 1) for i in range(Q["data"].shape[1])]
    df = pd.DataFrame(Q["data"], columns=feat_cols)
    if Q["method"] == 'pca':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(df)
    elif Q["method"] == 'tSNE':
        tsne = TSNE(n_components=2, method="exact", )
        Y = tsne.fit_transform(df)
    # TODO kernel pca

    x = Y[:, 0]
    y = Y[:, 1]
    n, n_sample = Q["data"].shape

    rval = minimum_bounding_rectangle(Y)
    for n in range(10):
        plt.scatter(Y[:, 0], Y[:, 1])
        bbox = minimum_bounding_rectangle(Y)
        plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
        plt.axis('equal')
        plt.show()
    plt.scatter(x, y)

    print(1)

#
#
# Y = tsne(Q.data, 'Algorithm', 'exact', 'Distance', Q.Dist);
