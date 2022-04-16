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
    print(data)
    data = dict(data)

    verts2d = np.array(data['verts2d'])
    vcolors = np.array(data['vcolors'])
    faces = np.array(data['faces'])
    depth = np.array(data['depth'])

    # Turn the image by 90 degrees
    verts2d_final = np.zeros((verts2d.shape[0], 2))
    verts2d_final[:, 0] = verts2d[:, 1]
    verts2d_final[:, 1] = verts2d[:, 0]

    return verts2d_final, vcolors, faces, depth


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
    theta = data['theta'][0]

    return C, F, V, O, H, W, M, N, w, g, ck, cu, cv, t1, t2, theta
