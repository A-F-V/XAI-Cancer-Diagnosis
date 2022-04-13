import matplotlib.pyplot as plt
import torch
from src.utilities.img_utilities import tensor_to_numpy


def plot_images(images, dimensions=(1, 1), cmap='nipy_spectral', figsize=None, **kwargs):
    """Plots the images passed as argument.
    Args:
        images: images to plot as a numpy array. The first dim should be the number of images. The second and third dims are the image dimensions. The fourth dim is the number of channels.
        dimensions: dimensions of the images.
    """
    images = list(map(lambda img: tensor_to_numpy(img) if torch.is_tensor(img) else img, images))
    if(len(images)) == 1:
        f = plt.figure(figsize=figsize)
        plt.imshow(images[0], cmap=cmap, **kwargs)
        plt.axis('off')
        return f
    f, ax = plt.subplots(dimensions[0], dimensions[1], figsize=(
        figsize if figsize != None else (dimensions[1]*2, dimensions[0]*2)))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            if min(dimensions) == 1:
                if max(i, j) >= len(images):
                    continue
                ax[max(i, j)].imshow(images[max(i, j)], cmap=cmap, **kwargs)
                ax[max(i, j)].axis('off')
            else:
                if i*dimensions[1]+j >= len(images):
                    continue
                ax[i, j].imshow(images[i*dimensions[1]+j], cmap=cmap, **kwargs)
                ax[i, j].axis('off')
    return f
