import numpy as np
import inc.Helpers.tools as tls
import inc.coloring as clr


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
                img[x, y] = clr.interpolate_color(int(np.around(x_left)), int(np.around(x_right)), x, c1, c2)

    return img


def render_flat(verts2d, vcolors, img):
    """Renders the image, using a single color for each triangle

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    img : MxNx3 image matrix

    Returns
    -------
    img : updated MxNx3 image matrix
    """
    new_color = np.array(np.mean(vcolors, axis=0))
    if (verts2d == verts2d[0]).all():
        img[int(verts2d[0, 0]), int(verts2d[0, 1])] = new_color
        return img

    # compute edge limits and sigma
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigma_of_edge = tls.compute_edge_limits(verts2d)

    # find min/max x and y
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))

    # find initial active edges for y = 0
    active_edges = np.array([False, False, False])
    active_nodes = np.zeros((3, 2))

    active_edges, active_nodes, is_invisible = tls.initial_active_elements(active_edges, active_nodes, vertices_of_edge,
                                                                           y_limits_of_edge, sigma_of_edge)
    if is_invisible:
        return img

    # dsp.show_vscan(y_min, active_edges, active_nodes, vertices_of_edge)
    for y in range(y_min, y_max + 1):
        # dsp.show_vscan(y, active_edges, active_nodes, vertices_of_edge)
        cross_counter = 0
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_nodes[active_edges][:, 0]))
            if cross_counter % 2 != 0:
                img[x, y] = new_color
            elif y == y_max and np.count_nonzero(x == np.around(active_nodes[active_edges][:, 0])) > 0:
                img[x, y] = new_color

        active_edges, active_nodes, updated_nodes = tls.update_active_edges(y, vertices_of_edge, y_limits_of_edge,
                                                                            sigma_of_edge, active_edges, active_nodes)
        active_nodes = tls.update_active_nodes(sigma_of_edge, active_edges, active_nodes, updated_nodes)
    return img


