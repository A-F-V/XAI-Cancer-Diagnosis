################################
# by @gatsby2016 from https://github.com/gatsby2016/Augmentation-PyTorch-Transforms, with modifications by myself
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
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """

    def __init__(self, theta=0.):
        assert isinstance(theta, numbers.Number), "theta should be a single number."
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

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ',alpha={0}'.format(self.alpha)
        format_string += ',betti={0}'.format(self.betti)
        return format_string


class RandomAffineCV2(object):
    """Random Affine transformation by CV2 method on image by alpha parameter.
    Args:
        alpha (float): alpha value for affine transformation
        mask (PIL Image) in __call__, if not assign, set None.
    """

    def __init__(self, alpha):
        assert isinstance(alpha, numbers.Number), "alpha should be a single number."
        assert 0. <= alpha <= 0.15, \
            "In pathological image, alpha should be in (0,0.15), you can change in myTransform.py"
        self.alpha = alpha

    @staticmethod
    def affineTransformCV2(img, alpha, masks=None):
        alpha = img.shape[1] * alpha
        if masks is not None:
            masks = np.array(masks).astype(np.uint8)
            img = np.concatenate((img, masks[..., None]), axis=2)

        imgsize = img.shape[:2]
        center = np.float32(imgsize) // 2
        censize = min(imgsize) // 3
        pts1 = np.float32([center+censize, [center[0]+censize, center[1]-censize], center-censize])  # raw point
        pts2 = pts1 + np.random.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)  # output point
        M = cv2.getAffineTransform(pts1, pts2)  # affine matrix
        img = cv2.warpAffine(img, M, imgsize[::-1],
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if masks is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.affineTransformCV2(np.array(img), self.alpha, mask)

    def __repr__(self):
        return self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)


class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
        # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
        # https://blog.csdn.net/maliang_1993/article/details/82020596
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
    Args:
        layers (List of Tensors): A list containing each of the image and mask layers
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
    """

    def __init__(self, alpha, sigma):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
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

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string
