import torch
from torchvision.transforms import Normalize as N
from torchvision.transforms import ToTensor as T


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        img = sample['image']
        return {'image': N(self.mean, self.std, inplace=True)(img), 'semantic_mask': sample['semantic_mask']}


class ToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return {prop: T()(sample[prop])for prop in sample}
