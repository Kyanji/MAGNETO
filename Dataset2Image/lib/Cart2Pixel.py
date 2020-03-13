import math
import pickle

import pandas as pd
import json as json
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from Dataset2Image.lib.MinRect import minimum_bounding_rectangle
from Dataset2Image.lib.ConvPixel import ConvPixel
import imageio
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import cv2
import matplotlib.pyplot as plt
import numpy as np


def Cart2Pixel(Q=None, A=None, B=None, dynamic_size=False):
    # TODO controls on input
    if A is not None:
        A = A - 1
    if (B != None):
        B = B - 1
    # to dataframe
    feat_cols = ["col-" + str(i + 1) for i in range(Q["data"].shape[1])]
    df = pd.DataFrame(Q["data"], columns=feat_cols)
    if Q["method"] == 'pca':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(df)
    elif Q["method"] == 'tSNE':
        tsne = TSNE(n_components=2, method="exact", )
        Y = tsne.fit_transform(df)
    elif Q["method"] == 'kpca':
        kpca = KernelPCA(n_components=2, kernel='linear')
        Y = kpca.fit_transform(df)

    x = Y[:, 0]
    y = Y[:, 1]
    n, n_sample = Q["data"].shape
    #plt.scatter(x, y)
    bbox = minimum_bounding_rectangle(Y)
    #plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
    # rotation
    grad = (bbox[1, 1] - bbox[0, 1]) / (bbox[1, 0] - bbox[0, 0])
    theta = np.arctan(grad)
    R = np.asmatrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    bboxMatrix = np.matrix(bbox)
    zrect = (R.dot(bboxMatrix.transpose())).transpose()
    # zrect=R.dot(bboxMatrix)
    #plt.fill(zrect[:, 0], zrect[:, 1], alpha=0.2)

    coord = np.array([x, y])
    rotatedData = np.array(R.dot(coord))  # Z
    # plt.scatter(rotatedData[0, :], rotatedData[1:])
    # plt.axis('square')

    #find duplicate
    for i in range(len(rotatedData[0, :])):
        for j in range(i + 1, len(rotatedData[0])):
            if rotatedData[0, i] == rotatedData[0, j] and rotatedData[1, i] == rotatedData[1, j]:
                print("duplicate:" + str(i) + " " + str(j))


    #nearest point

    min_dist = np.inf
    min_p1 = 0
    min_p2 = 0
    for p1 in range(n):
        for p2 in range(p1 + 1, n):
            d = (rotatedData[0, p1] - rotatedData[0, p2]) ** 2 + (rotatedData[1, p1] - rotatedData[1, p2]) ** 2
            if min_dist > d > 0 and p1 != p2:
                min_p1 = p1
                min_p2 = p2
                min_dist = d
    #plt.scatter([rotatedData[0, min_p1], rotatedData[0, min_p2]], [rotatedData[1, min_p1], rotatedData[1, min_p2]])
    # plt.show()

    # euclidean distance
    dmin = np.linalg.norm(rotatedData[:, min_p1] - rotatedData[:, min_p2])
    rec_x_axis = abs(zrect[0, 0] - zrect[1, 0])
    rec_y_axis = abs(zrect[1, 1] - zrect[2, 1])

    if dynamic_size:
        precision_old = math.sqrt(2)
        A = math.ceil(rec_x_axis * precision_old / dmin)
        B = math.ceil(rec_y_axis * precision_old / dmin)
        print("Dynamic [A:" + str(A) + " ; B:" + str(B) + "]")
        if max([A, B]) > Q["max_px_size"]:
            precision = precision_old * Q["max_px_size"] / max([A, B])
            A = math.ceil(rec_x_axis * precision / dmin)
            B = math.ceil(rec_y_axis * precision / dmin)
    # cartesian coordinates to pixels
    xp = np.round(
        1 + (A * (rotatedData[0, :] - min(rotatedData[0, :])) / (max(rotatedData[0, :]) - min(rotatedData[0, :]))))
    yp = np.round(
        1 + (-B) * (rotatedData[1, :] - max(rotatedData[1, :])) / (max(rotatedData[1, :]) - min(rotatedData[1, :])))
    A = max(xp)
    B = max(yp)

    images = []

    # Save Model to JSON file
    image_model = {"xp": xp.tolist(), "yp": yp.tolist(), "A": A, "B": B}
    j = json.dumps(image_model)
    f = open("dataset/CICDS2017/param/image_model.json", "w")
    f.write(j)
    f.close()

    images = []
    # Training set
    for i in range(0, 700):
        a=ConvPixel(Q["data"][:, i], xp, yp, A, B)
        plt.imshow(a, cmap="gray")
        input("pause")


    # images = [ConvPixel(Q["data"][:, i], xp, yp, A, B) for i in range(0, n_sample)]
        # images.append(ConvPixel(Q["data"][:, i], xp, yp, A, B))
        # filename = "dataset/CICDS2017/images/img" + str(i) + ".jpg"
        # cv2.imwrite(filename, images[i])
        # if i % 10000 == 0:
        #     print(str(i) + "of " + str(n_sample))
    #ret = json.dump([img.tolist() for img in images])

    filename = "dataset/CICDS2017/param/trainingsetImage.pickle"
    f_myfile = open(filename, 'wb')
    pickle.dump(images, f_myfile)
    f_myfile.close()


    return images, image_model
