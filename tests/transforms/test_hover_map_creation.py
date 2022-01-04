import numpy as np
from src.transforms.graph_construction.hover_maps import hover_map
import os

data_folder = os.path.join("tests", "data", "images")


def test_hover_map():
    mask = np.load(os.path.join(data_folder, "test_pannuke_instance_mask.npy"))
    h_map, v_map = hover_map(mask)
    assert h_map.min().item() >= -1
    assert v_map.min().item() >= -1
    assert h_map.max().item() <= 1
    assert v_map.max().item() <= 1
