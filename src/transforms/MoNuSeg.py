import torch
from torchvision.transforms import Normalize as N
from torchvision.transforms import ToTensor as T
from torchvision.transforms.functional import crop
from random import random
# Todo this this the way to do it? Is it bad to use dictionaries

# from torchvision import transforms, datasets


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        img = sample['image']
        output = {'image': N(self.mean, self.std, inplace=True)(img)}
        if 'semantic_mask' in sample:
            output['semantic_mask'] = sample['semantic_mask']
        return output


class ToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return {prop: T()(sample[prop])for prop in sample}


class RandomCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, sample):
        img = sample['image']
        dim = img.size()
        top, left = int(random()*(dim[1]-self.size[0])), int(random()*(dim[2]-self.size[1]))

        output = {prop: crop(sample[prop], top=top, left=left, height=self.size[0],
                             width=self.size[1]) for prop in sample}
        return output
