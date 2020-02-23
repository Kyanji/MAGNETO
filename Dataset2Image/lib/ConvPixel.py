import numpy as np
from collections import defaultdict


def ConvPixel(FVec, xp, yp, A, B, base, fig):
    n = len(FVec)
    M = np.ones([int(A), int(B)]) * base
    for j in range(0, n):
        M[int(xp[j]) - 1, int(yp[j]) - 1] = FVec[j]
    zp = np.array([xp, yp])

    zp[:, 0] = zp[:, 12]
    zp[:, 13] = zp[:, 0]
    zp[:, 15] = zp[:, 0]

    zp[:,6] = zp[:, 5]
    zp[:, 2] = zp[:, 6]
    zp[:, 11] = zp[:, 6]

    dup ={}
    # find duplicate
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if zp[0, i] == zp[0, j] and zp[1, i] == zp[1, j]:
                #if i in dup.keys():
                print("duplicate:" + str(i) + " " + str(j)+ "value: ")
                #dup.add(i)
                #dup[i].add(j)
                dup.setdefault(str(zp[0, i])+"-"+str(zp[1, i]), {i}).add(j)

    for index in dup.keys():
        x,y=index.split("-")
        M[int(float(x)) - 1, int(float(y)) - 1] = sum(FVec[list(dup[index])])/len(dup[index])
    print(1)
