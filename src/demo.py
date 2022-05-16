from inc.affine import affine_transform
from inc.transformation import TransformationMatrix
import inc.Helpers.reader as rd
from inc.renderer import render_object
import inc.Helpers.display as dsp
import inc.Helpers.tools as tls

img_H, img_W, M, N = 16, 16, 512, 512
f = 70

# load data, project 3D->2D and render the object
vcolors, faces, verts3d, u, ck, cu, cv, t1, t2, phi = rd.load_data_npy(filename='../data/hw2.npy')
normals = tls.calculate_normals(verts3d, faces)
print(normals)

# img1 = render_object(verts3d, faces, vcolors, M, N, img_H, img_W, f, cv, ck, cu)
# dsp.display_npy(img1, save=True, filename='1')
# transformation = TransformationMatrix()
#
# # Tranformation 1
# transformation.translate(t=t1)
# verts3d = affine_transform(cp=verts3d, T=transformation.T)
# img2 = render_object(verts3d, faces, vcolors, M, N, img_H, img_W, f, cv, ck, cu)
# dsp.display_npy(img2, save=True, filename='2')
#
# # Transformation 2
# transformation = TransformationMatrix()
# transformation.rotate(angle_radians=phi, rotation_axis=u)
# verts3d = affine_transform(cp=verts3d, T=transformation.T)
# img3 = render_object(verts3d, faces, vcolors, M, N, img_H, img_W, f, cv, ck, cu)
# dsp.display_npy(img3, save=True, filename='3')
#
# # Tranformation 3
# transformation = TransformationMatrix()
# transformation.translate(t=t2)
# verts3d = affine_transform(cp=verts3d, T=transformation.T)
# img4 = render_object(verts3d, faces, vcolors, M, N, img_H, img_W, f, cv, ck, cu)
# dsp.display_npy(img4, save=True, filename='4')
