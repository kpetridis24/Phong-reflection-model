import numpy as np


def system_transform(cp, T):
    """Computes the 3D coordinates of points, relative to a new coordinate system.

    Notes
    -----
    The new coordinate system is resulted from rotating the initial coordinate system (WCS). The points themselves
    remain static in space.

    Parameters
    ----------
    cp : Nx3 matrix containing the 3D coordinates of every point.
    T : the transformation matrix.

    Returns
    -------
    Nx3 matrix containing the 3D coordinates of the points, relative to the new coordinate system.
    """
    # turning the system with a matrix, is like turning all points with its inverse matrix. SO(3): inv = transp
    cp_augmented = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    dp_augmented = np.matmul(np.linalg.inv(T), cp_augmented)
    return dp_augmented.T[:, 0:len(dp_augmented) - 1]