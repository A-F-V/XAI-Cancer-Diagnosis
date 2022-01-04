import torch
from torchvision.transforms import Normalize as N
from torchvision.transforms import ToTensor as T
from torchvision.transforms.functional import crop, rotate, vflip, hflip, resize
from torchvision.transforms import InterpolationMode, ColorJitter
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
        output = {prop: torch.clip(sample[prop] + torch.normal(0, self.amt), 0, 1) if (self.fields ==
                                                                                       None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class ColourJitter(torch.nn.Module):
    def __init__(self, bcsh=(0, 0, 0, 0), fields=None):
        super().__init__()
        self.fields = fields
        self.bcsah = bcsh

    def forward(self, sample):
        output = {prop: ColorJitter(*self.bcsah)(sample[prop]) if (self.fields ==
                                                                   None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class Scale(torch.nn.Module):
    def __init__(self, x_fact, y_fact, img_field="image", fields=None):
        super().__init__()
        self.fields = fields
        self.x_fact = x_fact
        self.y_fact = y_fact

    def forward(self, sample):
        dim = sample[self.img_field].size()
        new_size = (int(dim[0]*self.x_fact), int(dim[0]*self.y_fact))
        output = {prop: resize(sample[prop], new_size) if (self.fields ==
                                                           None or prop in self.fields) else sample[prop] for prop in sample}
        return output


class RandomScale(torch.nn.Module):
    def __init__(self, x_fact_range, y_fact_range, img_field="image", fields=None):
        super().__init__()
        self.fields = fields
        self.x_fact_range = x_fact_range
        self.y_fact_range = y_fact_range

    def forward(self, sample):
        dim = sample[self.img_field].size()

        def random_in_range(rng):
            return (rng[1]-rng[0])*random()+rng[0]

        x_fact = random_in_range(self.x_fact_range)
        y_fact = random_in_range(self.y_fact_range)
        new_size = (int(dim[0]*x_fact), int(dim[0]*y_fact))
        output = {prop: resize(sample[prop], new_size) if (self.fields ==
                                                           None or prop in self.fields) else sample[prop] for prop in sample}
        return output


# todo blur