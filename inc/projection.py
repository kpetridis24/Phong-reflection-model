import numpy as np
from inc.coordinate_system import system_transform
from inc.transformation import TransformationMatrix


def project_camera(w, cv, cx, cy, cz, p):
    rotation_matrix = np.array([cx, cy, cz]).T
    M = TransformationMatrix()
    # matrix for operation R^-1 * cp + cv
    M.T[0:len(M.T) - 1, 0:len(M.T) - 1] = rotation_matrix
    M.T[0:len(M.T) - 1, len(M.T) - 1] = cv

    # coordinates of p, relative to the camera system
    p_camera = system_transform(p, M.T)
    m = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0]])

    p_camera_augmented = np.concatenate((p_camera.T, np.ones((1, p_camera.shape[0]))), axis=0)

    p_2D = np.matmul(m, - w * p_camera_augmented / p_camera_augmented[2])
    # P contains the 2D coordinates of every point (2xN). todo: Maybe P.T is not better.
    P = p_2D[0:2, :].T
    D = p_camera[:, 2]
    return P, D


def project_camera_ku(w, cv, c_lookat, c_up, p):
    zc = (c_lookat - cv) / np.linalg.norm(c_lookat - cv)    # pointing axis of camera
    t = c_up - np.dot(c_up, zc) * zc
    yc = t / np.linalg.norm(t)
    xc = np.cross(yc, zc)
    return project_camera(w, cv, xc, yc, zc, p)


