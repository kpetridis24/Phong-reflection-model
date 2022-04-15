import numpy as np


def convert_linear(old_min, old_max, new_min, new_max, old_value):
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


def rasterize(P, M, N, H, W):
    # convert every coordinate from the old range (H,W) to the new range (M,N)
    # P = [[x1, y1], [x2, y2], ... [xn, yn]]
    old_min_x, old_min_y, new_min_x, new_min_y = -W / 2, -H / 2, 0, 0
    old_max_x, old_max_y, new_max_x, new_max_y = W / 2 - 1, H / 2 - 1, N - 1, M - 1
    P_rast = np.array([convert_linear(old_min_x, old_max_x, new_min_x, new_max_x, P[:, 0]),
                       convert_linear(old_min_y, old_max_y, new_min_y, new_max_y, P[:, 1])]).T
    return np.around(P_rast - 0.5)