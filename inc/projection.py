import numpy as np
from inc.coordinate_system import system_transform
from inc.transformation import TransformationMatrix


def project_camera(w, cv, cx, cy, cz, p):
    rotation_matrix = np.array([cx, cy, cz]).T
    M = TransformationMatrix()
    # matrix for operation R^-1 * cp + cv
    M.T[0:len(M.T) - 1, 0:len(M.T) - 1] = rotation_matrix
    M.T[0:len(M.T) - 1, len(M.T) - 1] = cv
    M.T = np.around(M.T, 4)

    # coordinates of p, relative to the camera system
    P, D = np.zeros((p.shape[0], 2)), np.zeros((p.shape[0],))
    for i in range(p.shape[0]):
        cp_ = system_transform(np.array([p[i, :]]), M.T)
        cp_ = np.concatenate((cp_, np.array([np.ones((1,))])), axis=1)
        P[i, 0] = - (w * cp_[0, 0] / cp_[0, 2])
        P[i, 1] = - (w * cp_[0, 1] / cp_[0, 2])
        D[i] = cp_[0, 2]

    return P, D


def project_camera_ku(w, cv, c_lookat, c_up, p):
    zc = np.around((c_lookat - cv) / np.linalg.norm(c_lookat - cv), 4)    # pointing axis of camera
    t = c_up - np.dot(c_up, zc) * zc
    yc = np.around(t / np.linalg.norm(t), 4)
    xc = np.around(np.cross(yc, zc), 4)
    return project_camera(w, cv, xc, yc, zc, p)


