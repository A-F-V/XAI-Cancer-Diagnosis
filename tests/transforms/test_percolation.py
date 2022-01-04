import numpy as np
from src.transforms.graph_construction.percolation import island_identifier
import os


def test_island_identifier():
    #img = ((np.random.uniform(0, 1, (256, 256))*10) > 1).astype(np.uint8)
    #assert island_identifier(img, 256*256).shape == img.shape
    #assert island_identifier(img, 256*256).dtype == np.uint8
    #assert island_identifier(img, 256*256).max() > 1

    img2 = np.array([[1, 0, 0, 1, 1, 0, 0, 1, 1, 0]], dtype=np.uint8)
    assert island_identifier(img2).max() == 3

    img3 = np.load(os.path.join("tests", "data", "images", "test_island_identifier.npy"))
    assert island_identifier(img3).shape == img3.shape
    assert island_identifier(img3).max() == 13
