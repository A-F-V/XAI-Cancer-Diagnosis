import torch
from torchvision.transforms import Normalize as N
from torchvision.transforms import ToTensor as T
from torchvision.transforms.functional import crop, rotate, vflip, hflip, resize, gaussian_blur
from torchvision.transforms import InterpolationMode, ColorJitter
from src.transforms.image_processing.imported_augmentation import *
from src.utilities.img_utilities import tensor_to_numpy
from random import random
# Todo this this the way to do it? Is it bad to use dictionaries

# from torchvision import transforms, datasets


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, fields=None):
        """Normalizes an input image.

        Args:
            mean (dict): Field to Mean Tensor dictionary.
            std (dict): Field to Std Tensor dictionary.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        if fields is None:
            self.keys = list(mean.keys())
        else:
            self.keys = fields
        assert set(self.keys) == set(self.mean.keys()) == set(self.std.keys())

    def forward(self, sample):
        output = {}
        for key in sample.keys():
            if key in self.keys:
                output[key] = N(self.mean[key], self.std[key], inplace=True)(sample[key])
            else:
                output[key] = sample[key]
        return output


class ToTensor(torch.nn.Module):
    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields

    def forward(self, sample):
        return {prop: T()(sample[prop]) if (self.fields == None or prop in self.fields) else sample[prop] for prop in sample}


class RandomCrop(torch.nn.Module):
    def __init__(self, size, image_field='image', fields=None):
        super().__init__()
        self.size = size
        self.image_field = image_field
        self.fields = fields

    def forward(self, sample):
        img = sample[self.image_field]
        dim = img.size()
        top, left = int(random()*(dim[1]-self.size[0])), int(random()*(dim[2]-self.size[1]))

        output = {prop: crop(sample[prop], top=top, left=left, height=self.size[0],
                             width=self.size[1]) if (self.fields == None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomRotate(torch.nn.Module):
    def __init__(self, max_angle=360, fields=None):
        super().__init__()
        self.max_angle = max_angle
        self.fields = fields

    def forward(self, sample):
        angle = int(random()*self.max_angle)
        output = {prop: rotate(sample[prop], angle, interpolation=(InterpolationMode.BILINEAR if prop == "image" else InterpolationMode.NEAREST)) if (
            self.fields == None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomFlip(torch.nn.Module):
    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields

    def forward(self, sample):
        hf, vf = random() > 0.5, random() > 0.5

        def flip(img):
            output = img.clone()
            if hf:
                output = hflip(output)
            if vf:
                output = vflip(output)
            return output

        output = {prop: flip(sample[prop]) if (self.fields ==
                                               None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, amt, fields=None):
        super().__init__()
        self.fields = fields
        self.amt = amt

    def forward(self, sample):
        output = {prop: torch.clip(sample[prop] + torch.normal(torch.zeros_like(sample[prop]), torch.zeros_like(sample[prop])+self.amt), 0, 1) if (self.fields ==
                                                                                                                                                   None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class ColourJitter(torch.nn.Module):
    def __init__(self, bcsh=(0, 0, 0, 0), fields=None):
        super().__init__()
        self.fields = fields
        self.bcsh = bcsh

    def forward(self, sample):
        output = {prop: ColorJitter(*self.bcsh)(sample[prop]).clip(0, 1) if (self.fields ==
                                                                             None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class StainJitter(torch.nn.Module):
    def __init__(self, theta=0, fields=None):  # HED_light: theta=0.05; HED_strong: theta=0.2
        super().__init__()
        self.fields = fields
        self.theta = theta
        self.jitterer = HEDJitter(theta=self.theta)

    def forward(self, sample):
        output = {prop: self.jitterer(sample[prop]) if (self.fields ==
                                                        None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomElasticDeformation(torch.nn.Module):
    def __init__(self, alpha, sigma, fields=None):  # HED_light: theta=0.05; HED_strong: theta=0.2
        super().__init__()
        self.fields = fields
        self.alpha = alpha
        self.sigma = sigma
        self.deformer = RandomElastic(alpha=self.alpha, sigma=self.sigma)


#masks = np.array(masks).astype(np.uint8)
#img = np.concatenate((img, masks[..., None]), axis=2)


    def forward(self, sample):
        fields = self.fields if self.fields is not None else sample.keys()
        img__mask_layers = [sample[prop] for prop in fields]

        elastic_layers = self.deformer(img__mask_layers)
        named_layers = {prop: layer for prop, layer in zip(fields, elastic_layers)}
        output = {prop: named_layers[prop] if (prop in fields) else sample[prop] for prop in sample}
        return output


class Scale(torch.nn.Module):
    # todo docs should say either use new size or x&y factors
    # todo! rescaling should not add antialiasing to semantic and instance masks
    def __init__(self, new_size=None, x_fact=None, y_fact=None, img_field="image", modes={}, fields=None):
        super().__init__()
        self.fields = fields
        self.x_fact = x_fact
        self.y_fact = y_fact
        self.new_size = new_size
        self.modes = modes

    def forward(self, sample):
        dim = sample[self.img_field].size()

        new_size = (int(dim[0]*self.x_fact), int(dim[0]*self.y_fact)) if self.new_size is None else self.new_size
        output = {prop: resize(sample[prop], new_size, interpolation=(self.modes[prop] if prop in self.modes else InterpolationMode.NEAREST))
                  if (self.fields ==
                      None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomScale(torch.nn.Module):
    def __init__(self, x_fact_range, y_fact_range, img_field="image", modes={}, fields=None):
        super().__init__()
        self.fields = fields
        self.x_fact_range = x_fact_range
        self.y_fact_range = y_fact_range
        self.img_field = img_field
        self.modes = modes

    def forward(self, sample):
        dim = sample[self.img_field].size()

        def random_in_range(rng):
            return (rng[1]-rng[0])*random()+rng[0]

        x_fact = random_in_range(self.x_fact_range)
        y_fact = random_in_range(self.y_fact_range)
        new_size = (int(dim[1]*x_fact), int(dim[2]*y_fact))
        output = {prop: resize(sample[prop], new_size, interpolation=(self.modes[prop] if prop in self.modes else InterpolationMode.NEAREST))
                  if (self.fields ==
                      None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class GaussianBlur(torch.nn.Module):
    def __init__(self, amt=1, kernel=17, fields=None):
        super().__init__()
        self.fields = fields
        self.amt = amt
        self.kernel = kernel

    def forward(self, sample):
        output = {prop: gaussian_blur(sample[prop], self.kernel, self.amt) if (self.fields ==
                                                                               None or prop in self.fields) else sample[prop] for prop in sample}
        return output
