import numpy as np


def system_transform(cp, T):
    # turning the system with a matrix, is like turning all points with its inverse matrix. SO(3): inv = transp
    cp_augmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    dp_augmented = np.matmul(np.linalg.inv(T), cp_augmented)
    return dp_augmented.T[:, 0:len(dp_augmented) - 1]