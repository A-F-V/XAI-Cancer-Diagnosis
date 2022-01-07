import matplotlib.pyplot as plt


def plot_images(images, dimensions, cmap='nipy_spectral'):
    """Plots the images passed as argument.
    Args:
        images: images to plot as a numpy array. The first dim should be the number of images. The second and third dims are the image dimensions. The fourth dim is the number of channels.
        dimensions: dimensions of the images.
    """
    f, ax = plt.subplots(dimensions[0], dimensions[1], figsize=(dimensions[1]*2, dimensions[0]*2))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            if min(dimensions) == 1:
                if max(i, j) >= len(images):
                    continue
                ax[max(i, j)].imshow(images[max(i, j)], cmap=cmap)
                ax[max(i, j)].axis('off')
            else:
                if i*dimensions[1]+j >= len(images):
                    continue
                ax[i, j].imshow(images[i*dimensions[1]+j], cmap=cmap)
                ax[i, j].axis('off')
