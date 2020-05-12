import csv

import pandas as pd

ds = 3
train = 0
test = 0

if ds == 1:
    train = 20000/100*20  # attacchi
    test = 180000
elif ds == 2:
    train = 9067
    test = 119341
else:
    train = 79349
    test = 250436

def fix(f):
    a = list(f["TN_val"])
    b = list(f["FP_val"])
    c = list(f["FN_val"])
    d = list(f["TP_val"])
    f["TN_val"] = list(d)
    f["TP_val"] = list(a)
    f["FP_val"] = list(c)
    f["FN_val"] = list(b)
    return f


def fix_test(f):
    a = list(f["TN_test"])
    b = list(f["FP_test"])
    c = list(f["FN_test"])
    d = list(f["TP_test"])
    f["TN_test"] = list(d)
    f["TP_test"] = list(a)
    f["FP_test"] = list(c)
    f["FN_test"] = list(b)
    return f

def res(cm, val):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    if val and attacks[0] == train:
        print("ok")
    elif val:
        print("error val")
        return False,False
    if (not val) and attacks[0] == test:
        print("ok")
    elif not val:
        print("error")
        return False,False
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / 2
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [OA, AA, P, R, F1, FAR, TPR]
    return True,r


for dim in range(10, 11):
    for mode in ["CNN2"]:
        for collision in ["Mean", "MI"]:
            n = "C:/Users/deros/Desktop/res_" + str(dim) + "x" + str(dim) + "_" + collision + "_" + mode + ".csv "
            with open(n, 'r') as file:
                f = pd.DataFrame(list(csv.DictReader(file)))
            time = f["time"]
            del f['time']
            f = f.astype(float)

            cm_val = [[f["TP_val"], f["FN_val"]], [f["FP_val"], f["TN_val"]]]
            done,r = res(cm_val, True)
            if not done:
                f=fix(f)
                cm_val = [[f["TP_val"], f["FN_val"]], [f["FP_val"], f["TN_val"]]]
                done, r = res(cm_val, True)
            f["OA_val"] = r[0]
            f["P_val"] = r[2]
            f["R_val"] = r[3]
            f["F1_val"] = r[4]
            f["FAR_val"] = r[5]
            f["TPR_val"] = r[6]

            cm_test = [[f["TP_test"], f["FN_test"]], [f["FP_test"], f["TN_test"]]]
            done, r = res(cm_test, False)
            if not done:
                f=fix_test(f)
                cm_test = [[f["TP_test"], f["FN_test"]], [f["FP_test"], f["TN_test"]]]
                done, r = res(cm_test, False)


            f["OA_test"] = r[0]
            f["P_test"] = r[2]
            f["R_test"] = r[3]
            f["F1_test"] = r[4]
            f["FAR_test"] = r[5]
            f["TPR_test"] = r[6]
            f["time"] = time
            n = n.replace(".csv", "1.csv")
            print(n)
            f.to_csv(n, index=False)
