import numpy as np
import inc.Helpers.tools as tls
import inc.coloring as clr
import inc.light as lt


def render_phong(verts2d, center, vcolors, normals, camera_pos, ka, kd, ks, light_positions, light_intensities,
                 Ia, n_phong, img):
    if (verts2d == verts2d[0]).all():
        return img

    # compute edge limits and sigma
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigma_of_edge = tls.compute_edge_limits(verts2d)

    # find min/max x and y
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))

    # find initial active edges for y = 0
    active_edges = np.array([False, False, False])
    active_nodes = np.zeros((3, 2))

    node_combination_on_edge = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 2]
                                }

    active_edges, active_nodes, is_invisible = tls.initial_active_elements(active_edges, active_nodes, vertices_of_edge,
                                                                           y_limits_of_edge, sigma_of_edge)
    if is_invisible:
        return img

    active_normals = np.zeros((3, 3))

    for y in range(y_min, y_max):
        active_edges, active_nodes, updated_nodes = tls.update_active_edges(y, vertices_of_edge, y_limits_of_edge,
                                                                            sigma_of_edge, active_edges, active_nodes)
        active_nodes = tls.update_active_nodes(sigma_of_edge, active_edges, active_nodes, updated_nodes)

        img, active_nodes_color = tls.color_contour(y, node_combination_on_edge, x_limits_of_edge, y_limits_of_edge,
                                                    sigma_of_edge, active_edges, active_nodes, vcolors, img)
        for i, v in enumerate(active_nodes):
            if active_edges[i]:
                # The edge coordinates of every active node. This is the edge i.
                x_edge = np.array(x_limits_of_edge[i])
                y_edge = np.array(y_limits_of_edge[i])
                node_pair = node_combination_on_edge[i]
                n1, n2 = normals[node_pair[0]], normals[node_pair[1]]
                # case of horizontal edge
                if sigma_of_edge[i] == 0:
                    active_normals[i] = clr.interpolate_color(x_edge[0], x_edge[1], active_nodes[i, 0], n1, n2)
                    active_normals[i] = active_normals[i] / np.linalg.norm(active_normals[i])
                elif np.abs(sigma_of_edge[i]) == float('inf'):
                    active_normals[i] = clr.interpolate_color(y_edge[0], y_edge[1], y, n1, n2)
                    active_normals[i] = active_normals[i] / np.linalg.norm(active_normals[i])
                else:
                    active_normals[i] = clr.interpolate_color(y_edge[0], y_edge[1], y, n1, n2)
                    active_normals[i] = active_normals[i] / np.linalg.norm(active_normals[i])

        x_left, idx_left = np.min(active_nodes[active_edges, 0]), np.argmin(active_nodes[active_edges, 0])
        x_right, idx_right = np.max(active_nodes[active_edges, 0]), np.argmax(active_nodes[active_edges, 0])
        c1, c2 = active_nodes_color[active_edges][idx_left], active_nodes_color[active_edges][idx_right]
        n1, n2 = active_normals[active_edges][idx_left], active_normals[active_edges][idx_right]

        cross_counter = 0
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_nodes[active_edges, 0]))
            if cross_counter % 2 != 0:
                if x < img.shape[0] and x >= 0 and y < img.shape[1] and y >= 0:
                    new_normal = clr.interpolate_color(int(np.around(x_left)), int(np.around(x_right)), x, n1, n2)
                    new_normal = (new_normal / np.linalg.norm(new_normal))[0]
                    new_color = clr.interpolate_color(int(np.around(x_left)), int(np.around(x_right)), x, c1, c2)
                    img[x, y] = \
                        lt.ambient_light(ka, Ia) + \
                        lt.diffuse_light(center, new_normal, new_color, kd, light_positions, light_intensities) + \
                        lt.specular_light(center, new_normal, new_color, camera_pos, ks, n_phong, light_positions,
                                          light_intensities)
    return img


def render_smooth(verts2d, vcolors, img):
    """Renders the image, using interpolate colors to achieve smooth color transitioning

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    img : MxNx3 image matrix

    Returns
    -------
    img : updated MxNx3 image matrix
    """
    if (verts2d == verts2d[0]).all():
        return img

    # compute edge limits and sigma
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigma_of_edge = tls.compute_edge_limits(verts2d)

    # find min/max x and y
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))

    # find initial active edges for y = 0
    active_edges = np.array([False, False, False])
    active_nodes = np.zeros((3, 2))

    node_combination_on_edge = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 2]
                                }

    active_edges, active_nodes, is_invisible = tls.initial_active_elements(active_edges, active_nodes, vertices_of_edge,
                                                                           y_limits_of_edge, sigma_of_edge)
    if is_invisible:
        return img

    for y in range(y_min, y_max):
        active_edges, active_nodes, updated_nodes = tls.update_active_edges(y, vertices_of_edge, y_limits_of_edge,
                                                                            sigma_of_edge, active_edges, active_nodes)
        active_nodes = tls.update_active_nodes(sigma_of_edge, active_edges, active_nodes, updated_nodes)

        img, active_nodes_color = tls.color_contour(y, node_combination_on_edge, x_limits_of_edge, y_limits_of_edge,
                                                    sigma_of_edge, active_edges, active_nodes, vcolors, img)

        x_left, idx_left = np.min(active_nodes[active_edges, 0]), np.argmin(active_nodes[active_edges, 0])
        x_right, idx_right = np.max(active_nodes[active_edges, 0]), np.argmax(active_nodes[active_edges, 0])
        c1, c2 = active_nodes_color[active_edges][idx_left], active_nodes_color[active_edges][idx_right]

        cross_counter = 0
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_nodes[active_edges, 0]))
            if cross_counter % 2 != 0:
                if x < img.shape[0] and x >= 0 and y < img.shape[1] and y >= 0:
                    img[x, y] = clr.interpolate_color(int(np.around(x_left)), int(np.around(x_right)), x, c1, c2)

    return img
