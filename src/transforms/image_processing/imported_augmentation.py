################################
# Starting point from @gatsby2016 from https://github.com/gatsby2016/Augmentation-PyTorch-Transforms, but ultimately modified.
################################


from __future__ import division
from typing import List


import numpy as np
import numbers
from torchvision.transforms import ToTensor as T
from torch import Tensor
from PIL import Image, ImageFilter
from skimage import color
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from src.utilities.img_utilities import tensor_to_numpy, numpy_to_tensor


class HEDJitter(object):
    def __init__(self, theta=0.):
        self.theta = theta

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to Tensor image
        return T()(Image.fromarray(rsimg))

    def __call__(self, img: Tensor):
        theta = self.theta
        img_t = tensor_to_numpy(img)
        self.alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))
        return self.adjust_HED(img_t, self.alpha, self.betti)


class RandomElastic(object):
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(layers: List, alpha, sigma):
        shape = layers[0].shape
        img_size = shape[:2]
        alpha = shape[1] * alpha
        sigma = shape[1] * sigma

        dx = gaussian_filter((np.random.rand(*img_size) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*img_size) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        def morph(layer):
            cuts = [map_coordinates(layer[:, :, i], indices, order=0,
                                    mode='reflect').reshape((*shape[:2], 1)) for i in range(layer.shape[2])]
            return np.concatenate(cuts, axis=2)
        trans_layers = [numpy_to_tensor(morph(layer)) for layer in layers]

        return trans_layers

    def __call__(self, layers: List):

        n_layers = [layer.permute((1, 2, 0)).numpy() for layer in layers]
        return self.RandomElasticCV2(n_layers, self.alpha, self.sigma)
