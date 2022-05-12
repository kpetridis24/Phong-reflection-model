"""
Reader function to load data from .npy file
"""
import numpy as np
import scipy.io as io


def load_data_npy(filename):
    """Loads the necessary data from a numpy file.

    Parameters
    ----------
    filename : the name of the file, including the .npy extension.

    Returns
    -------
    verts3d : the 3D coordinates of all the vertices of the triangle
    vcolors : the RGB colors of the three vertices, of this triangle
    faces : the indices from the three vertices of every triangle
    u : the axis of rotation
    ck : the pointing coordinates of the camera
    cu : the upp-vector of the camera
    cv : the coordinates of the camera
    t1 : displacement transformation
    t2 : displacement transformation
    phi : the rotation angle in radians
    """
    data = np.load(filename, allow_pickle=True).tolist()
    data = dict(data)

    vcolors, faces, verts3d = np.array(data['vcolors']), np.array(data['faces']), np.array(data['verts3d'])
    u = data['u']
    ck, cu, cv = data['c_lookat'], data['c_up'], data['c_org']
    t1, t2 = data['t_1'], data['t_2']
    phi = data['phi']

    return vcolors, faces, verts3d, u, ck, cu, cv, t1, t2, phi


def load_data_mat(filename):
    """Loads the necessary data from a matlab file.

    Parameters
    ----------
    filename : the name of the file, including the .npy extension.

    Returns
    -------
    verts3d : the 3D coordinates of all the vertices of the triangle
    vcolors : the RGB colors of the three vertices, of this triangle
    faces : the indices from the three vertices of every triangle
    u : the axis of rotation
    ck : the pointing coordinates of the camera
    cu : the upp-vector of the camera
    cv : the coordinates of the camera
    t1 : displacement transformation
    t2 : displacement transformation
    phi : the rotation angle in radians
    """
    data = io.loadmat(filename)

    vcolors, faces, verts3d = np.array(data['C']), np.array(data['F'] - 1), np.array(data['V'])
    u = data['g'].T[0]
    ck, cu, cv = data['ck'].T[0], data['cu'].T[0], data['cv'].T[0]
    t1, t2 = data['t1'].T[0], data['t2'].T[0]
    phi = data['theta'][0]

    return vcolors, faces, verts3d, u, ck, cu, cv, t1, t2, phi
