import numpy as np


# todo: should P be the median point of the triangle?


def ambient_light(material, color, I_ambient):
    I = I_ambient * material.k_ambient
    return color * I


def diffuse_light(P, N, color, material, lights):
    # todo: check if N is unit vector. If not, divide by norm.
    I = np.zeros((len(lights), 3))
    for light in lights:
        L = (light.pos - P) / np.linalg.norm(light.pos - P)
        incidence_angle = np.dot(N, L)
        I += light.intensity * material.k_diffuse * incidence_angle
    return color * I[0]


def specular_light(P, N, color, camera_pos, material, lights):
    # todo: check if N is unit vector. If not, divide by norm.
    V = (camera_pos - P) / np.linalg.norm(camera_pos - P)
    I = np.zeros((len(lights), 3))
    for light in lights:
        L = (light.pos - P) / np.linalg.norm(light.pos - P)
        projection_LN = np.dot(N, L)
        cos_angle = np.dot(2 * N * projection_LN - L, V)
        I += light.intensity * material.k_specular * cos_angle
    return color * I[0]

# p = np.zeros((3,))
# n = np.array([0, 0, 1])
# c = np.array([0.2, 0.7, 0.5])
# pos = np.array([3, 4, 10])
# mat = PhongMaterial(0.2, 0.3, 0.5, 2)
# lts = PointLight(np.array([1, 1, 1]), np.array([0.3, 0.3, 0.6]))
