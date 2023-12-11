import numpy as np
from sklearn.metrics import mutual_info_score


def ConvPixel(FVec, xp, yp, A, B, base=1, custom_cut=None, index=0):
    n = len(FVec)
    M = np.ones([int(A), int(B)]) * base
    for j in range(0, n):
        M[int(xp[j]) - 1, int(yp[j]) - 1] = FVec[j]
    zp = np.array([xp, yp])

    dup = {}
    # find duplicate
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):

                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)

    if custom_cut is not None:
        M = np.delete(M, range(0, custom_cut), 0)
    return M
