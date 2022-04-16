import numpy as np


def rasterize(P, M, N, H, W):
    # convert every coordinate from the old range (H,W) to the new range (M,N)
    # P = [[x1, y1], [x2, y2], ... [xn, yn]]
    N_ = P.shape[0]
    P_rast = np.zeros((N_, 2))
    cver = M / H
    chor = N / W

    for i in range(N_):
        P_rast[i, 0] = np.around((P[i, 0] + H / 2) * cver - 0.5)
        P_rast[i, 1] = np.around((P[i, 1] + W / 2) * chor - 0.5)

    return P_rast