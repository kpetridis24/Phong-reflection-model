import numpy as np

class PhongMaterial:
    def __init__(self, ka, kd, ks, n):
        self.k_ambient = ka
        self.k_diffuse = kd
        self.k_specular = ks
        self.n_phong = n