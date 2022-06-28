import numpy as np

def ambient_light(k_ambient, I_ambient):
    I = k_ambient * I_ambient
    return I


def diffuse_light(P, N, color, k_diffuse, light_positions, light_intensities):
    L = ((light_positions - P) / np.linalg.norm(light_positions - P))[0]
    incidence_angle = np.dot(N, L)
    I = light_intensities * k_diffuse * incidence_angle
    return color * I


def specular_light(P, N, color, camera_pos, k_specular, n_phong, light_positions, light_intensities):
    V = (camera_pos - P) / np.linalg.norm(camera_pos - P)
    L = (light_positions - P) / np.linalg.norm(light_positions - P)
    projection_LN = np.dot(N, L)
    cos_angle = np.dot(2 * N * projection_LN - L, V)
    I = light_intensities * k_specular * cos_angle ** n_phong
    return color * I
