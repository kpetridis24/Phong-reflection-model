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
    verts2d_final : the coordinates of all the vertices of the triangle
    vcolors : the RGB colors of the three vertices, of this triangle
    faces : the indices from the three vertices of every triangle
    depth : the depth of every vertex
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
        verts2d_final : the coordinates of all the vertices of the triangle
        vcolors : the RGB colors of the three vertices, of this triangle
        faces : the indices from the three vertices of every triangle
        depth : the depth of every vertex
    """
    data = io.loadmat(filename)

    C, F, V, O = np.array(data['C']), np.array(data['F'] - 1), np.array(data['V']), np.array(data['O'])
    H, W, M, N = data['H'][0, 0], data['W'][0, 0], data['M'][0, 0], data['N'][0, 0]
    w, g = data['w'][0, 0], data['g'].T[0]
    ck, cu, cv = data['ck'].T[0], data['cu'].T[0], data['cv'].T[0]
    t1, t2 = data['t1'].T[0], data['t2'].T[0]
    phi = data['theta'][0]

    return C, F, V, g, ck, cu, cv, t1, t2, phi
