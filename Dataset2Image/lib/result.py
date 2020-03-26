import csv

import pandas as pd


def res(cm):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return r


with open('C:/Users/deros/Desktop/tesi doc/d2/res50x50_MI_CNN2.csv', 'r') as file:
    f = pd.DataFrame(list(csv.DictReader(file)))
time = f["time"]
del f['time']
f = f.astype(float)
cm_val = [[f["TN_val"], f["FP_val"]], [f["FN_val"], f["TP_val"]]]
r = res(cm_val)

f["OA_val"] = r[0]
f["P_val"] = r[2]
f["R_val"] = r[3]
f["F1_val"] = r[4]
f["FAR_val"] = r[5]
f["TPR_val"] = r[6]

cm_test = [[f["TN_test"], f["FP_test"]], [f["FN_test"], f["TP_test"]]]
r = res(cm_test)

f["OA_test"] = r[0]
f["P_test"] = r[2]
f["R_test"] = r[3]
f["F1_test"] = r[4]
f["FAR_test"] = r[5]
f["TPR_test"] = r[6]
f["time"] = time
f.to_excel('C:/Users/deros/Desktop/tesi doc/d2-Res/res50x50_MI_CNN2.xlsx', index=False)
