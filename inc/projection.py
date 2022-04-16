import numpy as np
from inc.coordinate_system import system_transform
from inc.transformation import TransformationMatrix


def project_camera(w, cv, cx, cy, cz, p):
    """Computes the projection of 3D points into the camera's curtain.

    Notes
    -----
    We basically have a set of points in 3D space, whose coordinates are relative to the WCS (world coordinate system).
    What this function does is, it computes the coordinates relative to a new coordinate system (camera system), which
    can be acquired if we rotate and translate the WCS. The steps to do that are, first we rotate the WCS to match the
    camera system (R * WCS). Rotating the coordinate system is equal to inverse rotating the points of the system. So
    to do this we perform a system transformation to all points. After that, we translate the beginning (0,0) of WCS to
    match the center of the camera, which means subtracting this translation from the coordinates of every point.
    Eventually, we get R^-1(cp) + cv, which is the transformation matrix that we use in order to apply all the above
    discussed transformations.

    Parameters
    ----------
    w : the distance between the lens and curtain of the camera.
    cv : the position vector of the camera.
    cx : x axis of the camera.
    cy : y axis of the camera, which is vertical.
    cz : z axis of the camera (pointing direction).
    p : Nx3 matrix containing the 3D coordinates of all points

    Returns
    -------
    P : Nx2 matrix containing the 2D coordinates, resulted from the projection from 3D to 2D.
    D : Nx1 matrix containing the depth of each point.
    """
    rotation_matrix = np.array([cx, cy, cz]).T
    # transformation matrix for both operations (rotation and translation).
    M = TransformationMatrix()
    M.T[0:len(M.T) - 1, 0:len(M.T) - 1] = rotation_matrix
    M.T[0:len(M.T) - 1, len(M.T) - 1] = cv
    M.T = np.around(M.T, 4)

    # coordinates of p, relative to the camera system.
    P, D = np.zeros((p.shape[0], 2)), np.zeros((p.shape[0],))
    for i in range(p.shape[0]):
        cp_ = system_transform(np.array([p[i, :]]), M.T)
        cp_ = np.concatenate((cp_, np.array([np.ones((1,))])), axis=1)
        P[i, 0] = - (w * cp_[0, 0] / cp_[0, 2])
        P[i, 1] = - (w * cp_[0, 1] / cp_[0, 2])
        D[i] = cp_[0, 2]

    return P, D


def project_camera_ku(w, cv, c_lookat, c_up, p):
    """Computes the projection of 3D points into the camera's curtain.

    Notes
    -----
    zc is the camera pointing axis
    yc is the vertical axis which is perpendicular to zc
    ax is perpendicular to both previous axes

    Parameters
    ----------
    w: the distance between the lens and curtain of the camera.
    cv: the position vector of the camera.
    c_lookat: the target at which the camera points to.
    c_up: the up-vector of the camera.
    p: Nx3 matrix containing the 3D coordinates of all points

    Returns
    -------
    @see : project_camera function
    """
    zc = np.around((c_lookat - cv) / np.linalg.norm(c_lookat - cv), 4)
    t = c_up - np.dot(c_up, zc) * zc
    yc = np.around(t / np.linalg.norm(t), 4)
    xc = np.around(np.cross(yc, zc), 4)
    return project_camera(w, cv, xc, yc, zc, p)


