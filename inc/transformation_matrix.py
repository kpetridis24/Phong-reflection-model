import numpy as np


class TransformationMatrix:
    # in augmented form
    def __init__(self):
        self.T = np.diag(np.ones(4, ))

    # in augmented form.
    def rotate(self, angle_degrees, rotation_axis):
        u = rotation_axis / np.linalg.norm(rotation_axis)
        theta = np.radians(angle_degrees)
        m1 = np.array([[u[0] ** 2, u[0] * u[1], u[0] * u[2]],
                       [u[0] * u[1], u[1] ** 2, u[1] * u[2]],
                       [u[0] * u[2], u[1] * u[2], u[2] ** 2]])
        m2 = np.diag(np.ones(3, ))
        m3 = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        rotation_matrix = (1 - np.around(np.cos(theta), decimals=2)) * m1 + np.around(np.cos(theta), decimals=2) * m2 \
                                                                          + np.around(np.sin(theta), decimals=2) * m3
        self.T[0:len(self.T) - 1, 0:len(self.T) - 1] = rotation_matrix

    # in augmented form
    def translate(self, t):
        self.T[0:len(self.T) - 1, len(self.T) - 1] = t