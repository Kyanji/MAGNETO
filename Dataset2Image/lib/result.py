import csv

import pandas as pd


def res(cm,val):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    if val and normals[0]==7400:
        print("ok")
    elif val:
        print("error val")
    if (not val) and normals[0]==56000:
        print("ok")
    elif not val:
        print("error")
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return r

for dim in range(8,11):
    for mode in ["CNN_Nature","CNN2"]:
        for collision in ["Mean","MI"]:
            n = "C:/Users/deros/Desktop/res_"+str(dim)+"x"+str(dim)+"_"+collision+"_"+mode+".csv"
            with open(n, 'r') as file:
                f = pd.DataFrame(list(csv.DictReader(file)))
            time = f["time"]
            del f['time']
            f = f.astype(float)
            # a=[list(f["TP_val"]), list(f["FN_val"]), list(f["FP_val"]), list(f["TN_val"])]
            #
            # f["TP_val"]=a[3]
            # f["FN_val"]=a[2]
            # f["FP_val"]=a[1]
            # f["TN_val"]=a[0]
            cm_val = [[f["TP_val"], f["FN_val"]], [f["FP_val"], f["TN_val"]]]

            r = res(cm_val,True)

            f["OA_val"] = r[0]
            f["P_val"] = r[2]
            f["R_val"] = r[3]
            f["F1_val"] = r[4]
            f["FAR_val"] = r[5]
            f["TPR_val"] = r[6]
            if "CNN2" in n:
                a=[list(f["TP_test"]), list(f["FN_test"]), list(f["FP_test"]), list(f["TN_test"])]
                f["TP_test"] = a[3]
                f["FN_test"] = a[2]
                f["FP_test"] = a[1]
                f["TN_test"] = a[0]

                cm_test = [[f["TP_test"], f["FN_test"]], [f["FP_test"], f["TN_test"]]]
                r = res(cm_test,False)
            else:
                cm_test = [[f["TP_test"], f["FN_test"]], [f["FP_test"], f["TN_test"]]]
                r = res(cm_test,False)

            f["OA_test"] = r[0]
            f["P_test"] = r[2]
            f["R_test"] = r[3]
            f["F1_test"] = r[4]
            f["FAR_test"] = r[5]
            f["TPR_test"] = r[6]
            f["time"] = time
            n = n.replace(".csv", ".xlsx")
            print(n)
            f.to_excel(n, index=False)
