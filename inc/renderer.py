import numpy as np
from inc.projection import project_camera_lookat
from inc.rasterize import rasterize
from inc.triangle_filling import render_smooth, render_phong
import inc.Helpers.tools as tls
import inc.light as lt


def render_object(focal, eye, lookat, up, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks,
                  n_phong, light_positions, light_intensities, Ia, shader):
    """Performs the necessary preprocessing, based on the light model chosen by the caller.

    Notes
    -----
    The preprocessing includes the calculation of the normal vectors on every triangle's vertex and the color computation
    on all points if gouraud shader is picked. If the phong shader is picked, only the normals are used to render the
    image.

    Parameters
    ----------
    focal: the distance between the camera lens and curtain
    eye: the position of the camera
    lookat: the pointing direction of the camera
    up: the up-vector of the camera
    M: length of image
    N: height of image
    H: length of camera curtain
    W: height of camera curtain
    verts: the 3D vertices
    vertex_colors: the color of every vertex
    face_indices: the vertices of every triangle
    ka: the ambient coefficient
    kd: the diffuse coefficient
    ks: the specular coefficient
    n_phong: the phong number
    light_positions: the position of the light source
    light_intensities: the intensity of the light source
    Ia: The ambient intensity
    shader: the light model of choice

    Returns
    -------
    MxNx3 colored image
    """
    normals = tls.calculate_normals(verts, face_indices)
    P, D = project_camera_lookat(focal, eye, lookat, up, verts)
    P_rast = rasterize(P, M, N, H, W)

    if shader == 'gouraud':
        for p, point in enumerate(verts):
            vertex_colors[p] = \
                lt.ambient_light(ka, Ia) + \
                lt.diffuse_light(point, normals[p], vertex_colors[p], kd, light_positions, light_intensities) + \
                lt.specular_light(point, normals[p], vertex_colors[p], eye, ks, n_phong, light_positions,
                                  light_intensities)

    img = render(verts, P_rast, face_indices, vertex_colors, D, M, N, normals, eye, ka, kd, ks, light_positions,
                 light_intensities, Ia, n_phong, shader)
    return img


def render(verts3d, verts2d, faces, vcolors, depth, m, n, normals, camera_pos, ka, kd, ks, light_positions,
           light_intensities, Ia, n_phong, shade_t):
    """Iterates over every triangle, and calls the coloring method for each triangle.

    Parameters
    ----------
    verts3d : Lx3 matrix containing the 3D coordinates of every vertex
    verts2d : Lx2 matrix containing the 2D coordinates of every vertex (L vertices)
    faces : Kx3 matrix containing the vertex indices of every triangle (K triangles)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    depth : Lx1 array containing the depth of every vertex in its initial, 3D scene
    m : length of image
    n : width of image
    normals : the normal vectors at every triangle's vertices
    camera_pos : the position of the camera
    ka : the ambient coefficient
    kd : the diffuse coefficient
    ks : the specular coefficient
    light_positions : the position of the light source
    light_intensities : the intensities of the light source
    Ia : the ambient intensity of the light source
    n_phong : the phong number
    shade_t : coloring strategy, with 'flat' and 'gouraud' indicating that every triangle should
    be filled with a single color and have a gradual color changing effect respectively

    Returns
    -------
    img : MxNx3 image with colors
    """
    assert shade_t in ('phong', 'gouraud') and m >= 0 and n >= 0
    img = np.ones((m, n, 3))
    # depth of every triangle. depth[i] = depth of triangle i
    triangles_depth = np.array(np.mean(depth[faces], axis=1))
    # order from the farthest triangle to the closest, depth-wise
    triangles_in_order = list(np.flip(np.argsort(triangles_depth)))

    for t in triangles_in_order:
        vertices_tr = faces[t]
        center_tr = np.mean(verts3d[vertices_tr, :], axis=0)
        verts2d_tr = np.array(verts2d[vertices_tr])  # x,y of the 3 vertices of triangle t
        vcolors_tr = np.array(vcolors[vertices_tr])  # color of the 3 vertices of triangle t
        normals_tr = normals[vertices_tr]

        if shade_t == 'phong':
            img = render_phong(verts2d_tr, center_tr, vcolors_tr, normals_tr, camera_pos, ka, kd, ks, light_positions,
                               light_intensities, Ia, n_phong, img)
        else:
            img = render_smooth(verts2d_tr, vcolors_tr, img)

    return img
