import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from inc.affine import affine_transform
from inc.coordinate_system import system_transform
from inc.transformation import TransformationMatrix
from inc.projection import project_camera, project_camera_ku
from inc.rasterize import rasterize
import inc.Helpers.reader as rd
from inc.renderer import render_object
import inc.Helpers.display as dsp


C, F, V, O, H, W, M, N, w, g, ck, cu, cv, t1, t2, theta = rd.load_data_mat(filename='../data/hw2.mat')
img = render_object(V, F, C, M, N, H, W, w, cv, ck, cu)
dsp.display_npy(img, save=True, filename='../results/img.png')
























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