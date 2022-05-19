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
    # print(data['point_light_pos'])
    Ia, focal, lookat = data['Ia'][0], data['focal'][0, 0], data['lookat'].T[0]
    normals, point_light_intensity, point_light_pos = data['normals'], data['point_light_intensity'][0], data[
        'point_light_pos'].T[0]
    M, N, W, H = data['M'][0, 0], data['N'][0, 0], data['W'][0, 0], data['H'][0, 0]
    vert_colors, face_indices, verts = np.array(data['vert_colors']), np.array(data['face_indices'] - 1), np.array(
        data['verts']).T
    eye, bg_color = data['eye'].T[0], data['bg_color'].T[0]
    ka, kd, ks = data['ka'][0, 0], data['kd'][0, 0], data['ks'][0, 0]
    n_phong, up = data['n_phong'][0, 0], data['up'].T[0]

    return vert_colors, face_indices, verts, M, N, W, H, focal, eye, up, lookat, ka, kd, ks, point_light_intensity, \
           point_light_pos, normals, bg_color, Ia, n_phong
