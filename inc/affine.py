import numpy as np


def affine_transform(cp, T):
    cp_augmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    cq_augmented = np.matmul(T, cp_augmented)
    return cq_augmented.T
