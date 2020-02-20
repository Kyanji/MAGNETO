import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lib.MinRect import minimum_bounding_rectangle
import matplotlib.pyplot as plt

import cv2
import numpy as np


def Cart2Pixel(Q=None):
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
    plt.scatter(x, y)
    bbox = minimum_bounding_rectangle(Y)
    plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
    plt.axis('equal')
    #rotation
    grad = (bbox[1,1]-bbox[0,1]) / (bbox[1,0]-bbox[0,0])
    theta = np.arctan(grad)
    R = np.asmatrix([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    bboxMatrix=np.matrix(bbox)
    zrect=(R.dot(bboxMatrix.transpose())).transpose()
    # zrect=R.dot(bboxMatrix)
    plt.fill(zrect[:, 0], zrect[:, 1], alpha=0.2)

    coord=np.array([x,y])
    rotatedData=np.array(R.dot(coord))
    plt.scatter(rotatedData[0,:],rotatedData[1:])
    plt.show()
    print(1)

#
#
# Y = tsne(Q.data, 'Algorithm', 'exact', 'Distance', Q.Dist);
