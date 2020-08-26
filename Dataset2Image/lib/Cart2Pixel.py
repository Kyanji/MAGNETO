import math
import pickle

import pandas as pd
import json as json
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from Dataset2Image.lib.MinRect import minimum_bounding_rectangle
from Dataset2Image.lib.ConvPixel import ConvPixel
import matplotlib.pyplot as plt
import imageio

plt.rcParams.update({'font.size': 22})

import numpy as np


def find_duplicate(zp):
    dup = {}
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):
                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)
    sum = 0
    for ind in dup.keys():
        sum = sum + (len(dup[ind]) - 1)
    return sum


def dataset_with_best_duplicates(X, y, zp):
    X = X.transpose()
    dup = {}
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):
                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)

    # print("Collisioni:" + str(len(dup.keys())))
    # print(dup.keys())
    toDelete = []
    for index in dup.keys():
        mi = []
        x_new = X[:, list(dup[index])]
        mi = mutual_info_classif(x_new, y)
        max = np.argmax(mi)
        dup[index].remove(list(dup[index])[max])
        toDelete.extend(list(dup[index]))
    X = np.delete(X, toDelete, axis=1)
    zp = np.delete(zp, toDelete, axis=1)
    return X.transpose(), zp, toDelete


def count_model_col(rotatedData, Q, r1, r2, params=None):
    tot = []
    for f in range(r1 - 1, r2):
        A = int(f * 2 / 3)
        B = f
        xp = np.round(
            1 + (A * (rotatedData[0, :] - min(rotatedData[0, :])) / (max(rotatedData[0, :]) - min(rotatedData[0, :]))))
        yp = np.round(
            1 + (-B) * (rotatedData[1, :] - max(rotatedData[1, :])) / (max(rotatedData[1, :]) - min(rotatedData[1, :])))
        zp = np.array([xp, yp])
        A = max(xp)
        B = max(yp)

        # find duplicates
        sum = str(find_duplicate(zp))
        print("Collisioni: " + sum)
        tot.append([A, B, sum])
        a = ConvPixel(Q["data"][:, 0], zp[0], zp[1], A, B)

        if params != None:
            plt.savefig(params["dir"] + str(A) + "x" + str(B) + '.png')
        else:
            plt.savefig(str(A) + '.png')
        plt.show()
    if params != None:
        pd.DataFrame(tot, columns=["indexA", "indexB", "collisions"]).to_excel(params["dir"] + "Collisionmezzi.xlsx",
                                                                               index=False)
    else:
        pd.DataFrame(tot, columns=["index", "collisions"]).to_excel("Collision.xlsx", index=False)


def Cart2Pixel(Q=None, A=None, B=None, dynamic_size=False, mutual_info=False, only_model=False, params=None):
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
        tsne = TSNE(n_components=2, method="exact")
        Y = tsne.fit_transform(df)
    elif Q["method"] == 'kpca':
        kpca = KernelPCA(n_components=2, kernel='linear')
        Y = kpca.fit_transform(df)

    x = Y[:, 0]
    y = Y[:, 1]
    n, n_sample = Q["data"].shape
    plt.scatter(x, y)
    bbox = minimum_bounding_rectangle(Y)
    plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
    # rotation
    grad = (bbox[1, 1] - bbox[0, 1]) / (bbox[1, 0] - bbox[0, 0])
    theta = np.arctan(grad)
    R = np.asmatrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    bboxMatrix = np.matrix(bbox)
    zrect = (R.dot(bboxMatrix.transpose())).transpose()
    # zrect=R.dot(bboxMatrix)
    plt.fill(zrect[:, 0], zrect[:, 1], alpha=0.2)

    coord = np.array([x, y])
    rotatedData = np.array(R.dot(coord))  # Z

    # rotatedData = np.delete(rotatedData, [59], 1)
    # rotatedData=np.delete(rotatedData, [175],1)
    # rotatedData=np.delete(rotatedData, [184],1)
    # Q["data"] = np.delete(Q["data"], [59], axis=0)
    # Q["data"] = np.delete(Q["data"], [175], axis=0)
    # Q["data"] = np.delete(Q["data"], [184], axis=0)
    # n = n - 1
    plt.scatter(rotatedData[0, :], rotatedData[1:])
    plt.axis('square')
    plt.show(block=False)

    # find duplicate
    for i in range(len(rotatedData[0, :])):
        for j in range(i + 1, len(rotatedData[0])):
            if rotatedData[0, i] == rotatedData[0, j] and rotatedData[1, i] == rotatedData[1, j]:
                print("duplicate:" + str(i) + " " + str(j))

    # nearest point

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
    # plt.scatter([rotatedData[0, min_p1], rotatedData[0, min_p2]], [rotatedData[1, min_p1], rotatedData[1, min_p2]])
    # plt.show(block=False)

    # euclidean distance
    dmin = np.linalg.norm(rotatedData[:, min_p1] - rotatedData[:, min_p2])
    rec_x_axis = abs(zrect[0, 0] - zrect[1, 0])
    rec_y_axis = abs(zrect[1, 1] - zrect[2, 1])
    if only_model:
        count_model_col(rotatedData, Q, 5, 50, params)

    if dynamic_size:
        precision_old = math.sqrt(2)
        A = math.ceil(rec_x_axis * precision_old / dmin)
        B = math.ceil(rec_y_axis * precision_old / dmin)
        print("Dynamic [A:" + str(A) + " ; B:" + str(B) + "]")
        if A > Q["max_A_size"] or B > Q["max_B_size"]:
            # precision = precision_old * Q["max_px_size"] / max([A, B])
            precision = precision_old * (Q["max_A_size"] / A) * (Q["max_B_size"] / B)
            A = math.ceil(rec_x_axis * precision / dmin)
            B = math.ceil(rec_y_axis * precision / dmin)
    # cartesian coordinates to pixels
    tot = []
    xp = np.round(
        1 + (A * (rotatedData[0, :] - min(rotatedData[0, :])) / (max(rotatedData[0, :]) - min(rotatedData[0, :]))))
    yp = np.round(
        1 + (-B) * (rotatedData[1, :] - max(rotatedData[1, :])) / (max(rotatedData[1, :]) - min(rotatedData[1, :])))
    # Modified Feature Position | custom cut
    cut = params["cut"]
    if cut is not None:
        xp[59] = cut
    zp = np.array([xp, yp])
    A = max(xp)
    B = max(yp)

    # find duplicates
    print("Collisioni: " + str(find_duplicate(zp)))

    # Training set

    images = []
    toDelete = 0
    name = "_" + str(int(A)) + 'x' + str(int(B))
    if params["No_0_MI"]:
        name = name + "_No_0_MI"
    if mutual_info:
        Q["data"], zp, toDelete = dataset_with_best_duplicates(Q["data"], Q["y"], zp)
        name = name + "_MI"
    else:
        name = name + "_Mean"
    if cut is not None:
        name = name + "_Cut" + str(cut)

    # save model to file
    image_model = {"xp": zp[0].tolist(), "yp": zp[1].tolist(), "A": A, "B": B, "custom_cut": cut,
                   "toDelete": toDelete}
    j = json.dumps(image_model)
    f = open(params["dir"] + "model" + name + ".json", "w")
    f.write(j)
    f.close()

    if only_model:
        a = ConvPixel(Q["data"][:, 0], zp[0], zp[1], A, B)
        plt.imshow(a, cmap="gray")
        plt.show()
    else:  # custom_cut=range(0, cut),
        a = ConvPixel(Q["data"][:, i], zp[0], zp[1], A, B, index=i)
        plt.imshow(a, cmap="gray")
        plt.show()
        if cut is not None:
            images = [ConvPixel(Q["data"][:, i], zp[0], zp[1], A, B, custom_cut=cut - 1, index=i) for i in
                      range(0, n_sample)]
        else:
            # a=np.where(Q["y"]==0)
            # attacks=Q["data"][:,a]
            for i in range(0, 3):
                images = [ConvPixel(Q["data"][:, i], zp[0], zp[1], A, B, index=i) for i in
                          range(0, n_sample)]

        filename = params["dir"] + "train" + name + ".pickle"
        f_myfile = open(filename, 'wb')
        pickle.dump(images, f_myfile)
        f_myfile.close()

    return images, image_model, toDelete
