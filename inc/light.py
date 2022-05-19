import numpy as np


# todo: should P be the median point of the triangle?


def ambient_light(material, color, I_ambient):
    I = I_ambient * material.k_ambient
    return color * I


def diffuse_light(P, N, color, material, lights):
    I = np.zeros((len(lights), 3))
    for light in lights:
        L = (light.pos - P) / np.linalg.norm(light.pos - P)
        incidence_angle = np.dot(N, L)
        I *= light.intensity * material.k_diffuse * incidence_angle
    return color * I[0]


def specular_light(P, N, color, camera_pos, material, lights):
    V = (camera_pos - P) / np.linalg.norm(camera_pos - P)
    I = np.zeros((len(lights), 3))
    for light in lights:
        L = (light.pos - P) / np.linalg.norm(light.pos - P)
        projection_LN = np.dot(N, L)
        cos_angle = np.dot(2 * N * projection_LN - L, V)
        I *= light.intensity * material.k_specular * cos_angle
    return color * I[0]
