import numpy as np


def ambient_light(k_ambient, I_ambient):
    """Computes the ambient component of the Phong lighting model

    Parameters
    ----------
    k_ambient: the ambient coefficient which is a property of the material
    I_ambient: environment property, related to the intensity of the light source

    Returns
    -------
    The ambient component to be added to the pixel of interest
    """
    I = k_ambient * I_ambient
    return I


def diffuse_light(P, N, color, k_diffuse, light_positions, light_intensities):
    """Computes the diffuse component of the Phong lighting model

    Parameters
    ----------
    P: the position of the point being rendered
    N: the normal vector at the point of interest
    color: the initial color of the point
    k_diffuse: the diffuse coefficient
    light_positions: the position of the light source
    light_intensities: the intensity of the light source

    Returns
    -------
    The diffuse component to be added to the pixel of interest
    """
    L = ((light_positions - P) / np.linalg.norm(light_positions - P))
    incidence_angle = np.dot(N, L)
    I = light_intensities * k_diffuse * incidence_angle
    return color * I


def specular_light(P, N, color, camera_pos, k_specular, n_phong, light_positions, light_intensities):
    """Computes the specular component of the Phong lighting model

    Parameters
    ----------
    P: the position of the point being rendered
    N: the normal vector at the point of interest
    color: the initial color of the point
    camera_pos: the position of the camera
    k_specular: the specular coefficient
    n_phong: the phong number
    light_positions: the position of the light source
    light_intensities: the intensity of the light source

    Returns
    -------
    The specular component to be added to the pixel of interest
    """
    V = (camera_pos - P) / np.linalg.norm(camera_pos - P)
    L = (light_positions - P) / np.linalg.norm(light_positions - P)
    projection_LN = np.dot(N, L)
    cos_angle = np.dot(2 * N * projection_LN - L, V)
    I = light_intensities * k_specular * cos_angle ** n_phong
    return color * I
