from inc.affine import affine_transform
from inc.transformation import TransformationMatrix
import inc.Helpers.reader as rd
from inc.renderer import render_object
import inc.Helpers.display as dsp


# load data, project 3D->2D and render the object
C, F, V, O, H, W, M, N, w, g, ck, cu, cv, t1, t2, theta = rd.load_data_mat(filename='../data/hw2.mat')
img1 = render_object(V, F, C, M, N, H, W, w, cv, ck, cu)
dsp.display_npy(img1, save=True, filename='1')
# Transformation matrix
transformation = TransformationMatrix()

# Tranformation 1
transformation.translate(t=t1)
V = affine_transform(cp=V, T=transformation.T)
img2 = render_object(V, F, C, M, N, H, W, w, cv, ck, cu)
dsp.display_npy(img2, save=True, filename='2')

# Transformation 2
transformation.rotate(angle_radians=theta, rotation_axis=g)
V = affine_transform(cp=V, T=transformation.T)
img3 = render_object(V, F, C, M, N, H, W, w, cv, ck, cu)
dsp.display_npy(img3, save=True, filename='3')

# Tranformation 3
transformation.translate(t=t2)
V = affine_transform(cp=V, T=transformation.T)
img4 = render_object(V, F, C, M, N, H, W, w, cv, ck, cu)
dsp.display_npy(img4, save=True, filename='4')
