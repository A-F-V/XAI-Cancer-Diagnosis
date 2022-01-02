import torch
from torchvision.transforms import Normalize as N
from torchvision.transforms import ToTensor as T
from torchvision.transforms.functional import crop, rotate
from torchvision.transforms import InterpolationMode
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
            self.keys = mean.keys()
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
        self.img_field = image_field
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
        output = {prop: rotate(sample[prop], angle, Interpolationmode=InterpolationMode.BILINEAR) if (
            self.fields == None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomFlip(torch.nn.Module):
    def __init__(self, fields=None):
        super().__init__()
        self.image_field = image_field
        self.fields = fields

    def forward(self, sample):
        img = sample[self.image_field]
        dim = img.size()
        output = {prop: img.flip(int(random()*dim[0])) if (self.fields ==
                                                           None or prop in self.fields) else sample[prop] for prop in sample}
        return output
