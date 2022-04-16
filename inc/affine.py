import numpy as np


def affine_transform(cp, T):
    """Applies an Affine Transformation on 3D coordinates

    Parameters
    ----------
    cp : Nx3 matrix, containing the 3D coordinates of every point
    T : The transformation matrix

    Returns
    -------
    Nx3 matrix containing the transformed coordinates
    """
    cp_augmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    cq_augmented = np.matmul(T, cp_augmented)
    return cq_augmented.T[:, 0:3]
