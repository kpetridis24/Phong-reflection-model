import numpy as np


def interpolate_color(x1, x2, x, c1, c2):
    """Computes the RGB color of a point, via the linear interpolation of colors from two triangle vertices

    At this point we assume that the choice of whether to use the x or y of the two vertices has already been done. This
    means that whether the two vertices belong to the same x or y has already been checked before calling this function,
    and the appropriate arguments (x or y) have been passed. If the vertices belong to a horizontal line, then x should
    be passed, in all other cases (including a vertical line), y is passed.

    Parameters
    ----------
    x1 : x or y of the first triangle vertex
    x2 : x or y of the second triangle vertex
    x : position to calculate the interpolation
    c1 : RGB color of the first vertex
    c2 : RGB color of the second vertex

    Returns
    -------
    color_value : 1x3 vector, containing the RGB color of the specified position
    """
    sigma = (x2 - x) / (x2 - x1)
    color_value = np.array([sigma * c1 + (1 - sigma) * c2])
    return color_value