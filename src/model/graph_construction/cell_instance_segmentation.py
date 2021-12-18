from pickletools import markobject
import numpy as np
from skimage.segmentation import watershed
from skimage.feature.peak import peak_local_max
import matplotlib.pyplot as plt
from scipy import ndimage

# todo look into dilation and erosion
# todo understand better the parameters of each of these functions


def instance_segment(image: np.ndarray, peak_seperation: int = 2, min_rel_threshold: float = 0.3,):
    """Takes a binary mask representing the segmented H&E image, and identifies the individual nuclei.

    Args:
        image (np.ndarray): The binary mask
    """
    image = ndimage.binary_dilation(image, iterations=3)
    distance = ndimage.distance_transform_edt(image)
    terrain = -distance          # creates terrain. Want to turn hills to valleys so that we can fill with water and find watersheds

    mask = peak_local_max(distance, footprint=np.ones((5, 5)), labels=image,
                          indices=False, min_distance=peak_seperation, threshold_rel=min_rel_threshold)
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=image)
    #instance_mask = watershed(terrain, mask=image)
    return labels
