import numpy as np
from inc.projection import project_camera_lookat
from inc.rasterize import rasterize
from inc.triangle_filling import render_flat, render_smooth
import inc.Helpers.tools as tls
import inc.light as lt


def render(verts2d, faces, vcolors, depth, m, n, shade_t):
    """Iterates over every triangle, and calls the coloring method for each triangle.

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    faces : Kx3 matrix containing the vertex indices of every triangle (K triangles)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    depth : Lx1 array containing the depth of every vertex in its initial, 3D scene
    m : length of image
    n : width of image
    shade_t : coloring strategy, with 'flat' and 'gouraud' indicating that every triangle should
    be filled with a single color and have a gradual color changing effect respectively

    Returns
    -------
    img : MxNx3 image with colors
    """
    assert shade_t in ('flat', 'gouraud') and m >= 0 and n >= 0
    img = np.ones((m, n, 3))
    # depth of every triangle. depth[i] = depth of triangle i
    depth_tr = np.array(np.mean(depth[faces], axis=1))
    # order from the farthest triangle to the closest, depth-wise
    triangles_in_order = list(np.flip(np.argsort(depth_tr)))

    for t in triangles_in_order:
        vertices_tr = faces[t]
        verts2d_tr = np.array(verts2d[vertices_tr])  # x,y of the 3 vertices of triangle t
        vcolors_tr = np.array(vcolors[vertices_tr])  # color of the 3 vertices of triangle t
        if shade_t == 'flat':
            img = render_flat(verts2d_tr, vcolors_tr, img)
        else:
            img = render_smooth(verts2d_tr, vcolors_tr, img)

    return img


def render_object(p, faces, vcolors, M, N, img_H, img_W, f, cv, c_lookat, c_up):
    """Projects a 3D object into 2D space and displays it on image.

    Parameters
    ----------
    p : Nx3 matrix containing the 3D coordinates of all points
    faces : Kx3 matrix containing the vertex indices of every triangle (K triangles).
    vcolors : Nx3 matrix containing the RGB colors of every point.
    M : length of the image frame.
    N : width of the image frame.
    img_H : length of the camera's curtain.
    img_W : width of the camera's curtain
    f : the distance between the lens and curtain of the camera.
    cv : the position vector of the camera.
    c_lookat : the target at which the camera points to.
    c_up : the up-vector of the camera.

    Returns
    -------
    img : MxNx3 image matrix
    """
    P, D = project_camera_lookat(f, cv, c_lookat, c_up, p)
    P_rast = rasterize(P, M, N, img_H, img_W)

    img = render(verts2d=P_rast, faces=faces, vcolors=vcolors, depth=D, m=M, n=N, shade_t='gouraud')

    return img


def render_object2(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, verts_colors, face_indices, mat, lights,
                   Ia):
    normals = tls.calculate_normals(verts, face_indices)
    P, D = project_camera_lookat(focal, eye, lookat, up, verts)
    P_rast = rasterize(P, M, N, H, W)

    img = np.ones((M, N, 3))
    # depth of every triangle. depth[i] = depth of triangle i
    depth_tr = np.array(np.mean(D[face_indices], axis=1))
    # order from the farthest triangle to the closest, depth-wise
    triangles_in_order = list(np.flip(np.argsort(depth_tr)))

    for p, point in enumerate(verts):
        verts_colors[p] += lt.ambient_light(mat, verts_colors[p], Ia) + \
                           lt.diffuse_light(point, normals[p], verts_colors[p], mat, lights) + \
                           lt.specular_light(point, normals[p], verts_colors[p], eye, mat, lights)

    for t in triangles_in_order:
        vertices_tr = face_indices[t]
        verts2d_tr = np.array(P_rast[vertices_tr])  # x,y of the 3 vertices of triangle t
        vcolors_tr = np.array(verts_colors[vertices_tr])  # color of the 3 vertices of triangle t
        # normals_tr = np.array(normals[vertices_tr])
        #     # img = render_flat(verts2d_tr, vcolors_tr, img)
        img = render_flat(verts2d_tr, vcolors_tr, img)
    #     shade_gouraud(verts2d_tr, normals_tr, vcolors_tr, None, eye, mat, lights, Ia, img)

    return img

# def shade_gouraud(verts_p, verts_n, verts_c, bcoords, camera_pos, mat, lights, Ia, X):
#     for p, point in enumerate(verts_p):
#         verts_c[p] = lt.ambient_light(mat, verts_c[p], Ia) + \
#                      lt.diffuse_light(point, verts_n[p], verts_c[p], mat, lights) + \
#                      lt.specular_light(point, verts_n[p], verts_c, camera_pos, mat, lights)
#         print(verts_c[p])
