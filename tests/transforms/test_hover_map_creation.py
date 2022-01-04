import numpy as np
from src.transforms.graph_construction.hover_maps import hover_map
import os

test_data_folder = os.path.join("tests", "data", "images")
data_folder = os.path.join("data", "processed", "PanNuke")


def test_hover_map():
    mask = np.load(os.path.join(test_data_folder, "test_pannuke_instance_mask.npy"))
    h_map, v_map = hover_map(mask)
    assert h_map.min().item() >= -1
    assert v_map.min().item() >= -1
    assert h_map.max().item() <= 1
    assert v_map.max().item() <= 1

    masks = np.load(os.path.join(data_folder, "masks.npy"), mmap_mode='r+')
    curious_mask = masks[4]
    h_map, v_map = hover_map(curious_mask)
    assert h_map.min().item() >= -1
    assert v_map.min().item() >= -1
    assert h_map.max().item() <= 1
    assert v_map.max().item() <= 1
