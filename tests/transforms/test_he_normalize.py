from src.transforms.image_processing.he_normalize import *
import pytest
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor


@pytest.fixture
def img14():
    img_path = os.path.join(os.getcwd(), 'data', 'processed', 'MoNuSeg_TRAIN', 'images', '14.tif')
    img = Image.open(img_path)
    return ToTensor()(img)


def test_get_stains(img14):
    assert get_stain_vectors(img14).shape == (2, 3)
