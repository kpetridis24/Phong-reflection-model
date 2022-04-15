import numpy as np
import matplotlib.pyplot as plt
from inc.affine import affine_transform
from inc.coordinate_system import system_transform
from inc.transformation import TransformationMatrix
from inc.projection import project_camera, project_camera_ku
from inc.rasterize import rasterize


H, W = 120, 120
M, N = 512, 512

# cp = np.array([[1, 2, 2]])
cp = np.array([[2, 2, 3], [0, 1, 2], [56, 7, 23]])

P, D = project_camera_ku(1, np.array([1, 1, 1]), np.array([2, 3, 2]), np.array([1, 2, 2]), cp)
print(P)

P_rast = rasterize(P, M, N, H, W)
print(P_rast)
























# M = TransformationMatrix()
# M.rotate(angle_degrees=48, rotation_axis=np.array([1, 4, 3]))
# print(M.T)


#
# # cq = affine_transform(cp, M.T)
# # print(cq)
#
# # cq = system_transform(cp, M.T)
# # print(cq)
#
# # P, D = project_camera(1, np.array([1, 1, 1]), np.array([1, 2, 2]), np.array([2, 1, 1]), np.array([3, 3, 2]), cp)
#
# P, D = project_camera_ku(1, np.array([1, 1, 1]), np.array([2, 3, 2]), np.array([1, 2, 2]), cp)
#
# print(P)
# print(D)