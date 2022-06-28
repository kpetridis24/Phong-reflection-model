import inc.Helpers.reader as rd
from inc.renderer import render_object
import inc.Helpers.display as dsp

focal = 70

verts, vertex_colors, face_indices, eye, up, lookat, ka, kd, ks, n_phong, \
light_positions, light_intensities, M, N, W, H, Ia = rd.load_data_npy(filename='../data/h3.npy')

img = render_object(focal, eye, lookat, up, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n_phong,
                    light_positions, light_intensities, Ia, 'gouraud')
dsp.display_npy(img, save=False, filename='gouraud')

img2 = render_object(focal, eye, lookat, up, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n_phong,
                     light_positions, light_intensities, Ia, 'phong')
dsp.display_npy(img2, save=False, filename='phong')
