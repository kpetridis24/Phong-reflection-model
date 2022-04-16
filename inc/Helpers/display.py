import matplotlib.pyplot as plt
import imageio
import numpy as np


def display_npy(img, save=False, filename='out'):
    """Displays a numpy matrix as a PNG image

    Parameters
    ----------
    img : MxNx3 image with RGB colors
    save : indicates whether to save the image
    filename : the name to store the image, without the extension
    """
    plt.imshow(img, interpolation='nearest')
    plt.show()
    if save:
        imageio.imsave('../results/' + filename + '.png', (img * 255).astype(np.uint8))
