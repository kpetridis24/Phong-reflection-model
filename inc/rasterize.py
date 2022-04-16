import numpy as np


def rasterize(P, M, N, H, W):
    """Places the 2D points from the camera's curtain into the image's frame.

    Parameters
    ----------
    P : Nx2 matrix containing the 2D coordinates of every point.
    M: length of the image frame.
    N: width of the image frame.
    H: length of the camera's curtain.
    W: width of the camera's curtain

    Returns
    -------
    P_rast : Nx2 matrix containing the coordinates of the points on the image frame.
    """
    num_points = P.shape[0]
    P_rast = np.zeros((num_points, 2))
    vertical = M / H
    horizontal = N / W

    for i in range(num_points):
        P_rast[i, 0] = np.around((P[i, 0] + H / 2) * vertical - 0.5)
        P_rast[i, 1] = np.around((P[i, 1] + W / 2) * horizontal - 0.5)

    return P_rast