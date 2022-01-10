from src.datasets.MoNuSeg import MoNuSeg
import os
from src.transforms.image_processing.augmentation import *
from torchvision.transforms import Compose, RandomChoice, RandomApply

scale_modes = {"image": InterpolationMode.BILINEAR,
               "semantic_mask": InterpolationMode.NEAREST, "instance_map": InterpolationMode.NEAREST}
transforms_training = Compose([
    Compose([
        RandomChoice([
            RandomScale(x_fact_range=(0.25, 0.35), y_fact_range=(0.25, 0.35),
                        modes=scale_modes),
            RandomScale(x_fact_range=(0.45, 0.55), y_fact_range=(0.45, 0.55),
                        modes=scale_modes),
            RandomScale(x_fact_range=(0.65, 0.75), y_fact_range=(0.65, 0.75),
                        modes=scale_modes),
            RandomScale(x_fact_range=(0.95, 1.05), y_fact_range=(0.95, 1.05),
                        modes=scale_modes),

        ], p=(0.15, 0.15, 0.2, 0.5)),
        RandomCrop(size=(64, 64))
    ]),
    RandomApply(
        [
            RandomFlip(),
            AddGaussianNoise(0.01, fields=["image"]),
            ColourJitter(bcsh=(0.2, 0.1, 0.1, 0.01), fields=["image"])
        ],

        p=0.5),
    Normalize(
        {"image": [0.6441, 0.4474, 0.6039]},
        {"image": [0.1892, 0.1922, 0.1535]})
])


def test_MoNuSeg_load_transform():
    dataset = MoNuSeg(os.path.join("data", "processed", "MoNuSeg_TRAIN"), transform=transforms_training)
    dataset[0]
    dataset = MoNuSeg(os.path.join("data", "processed", "MoNuSeg_TEST"), transform=transforms_training)
    dataset[0]
