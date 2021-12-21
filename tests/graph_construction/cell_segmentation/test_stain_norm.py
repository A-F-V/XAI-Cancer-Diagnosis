import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
from src.transforms.he_normalize import get_stain_vectors


def test_stain_norm_writeoutput():
    img_path = os.path.join("data", "processed", "MoNuSeg_TRAIN", "images", "0.tif")
    img = to_tensor(Image.open(img_path))
    res = get_stain_vectors(img)
    assert tuple(res[0].squeeze().shape) == (3,)
    assert tuple(res[1].squeeze().shape) == (3,)
