import numpy as np


class PointLight:
    def __init__(self, pos, intensity):
        self.pos = pos  # 3x1 vector of coordinates
        self.intensity = intensity  # 3x1 vector of intensity
