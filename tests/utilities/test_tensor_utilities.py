from src.utilities.tensor_utilties import *
import numpy as np
from torch import Tensor


def test_map_value():
    narr1 = np.array([1, 2, 3, 10, 0, 5, 10], dtype=np.uint8)
    res1 = map_value_numpy_array(narr1, 10, 0)
    assert res1.tolist() == [1, 2, 3, 0, 0, 5, 0]
    assert res1.dtype == np.uint8

    narr2 = np.array([[1.5, 2.5, 12]], dtype=np.float32)
    assert map_value_numpy_array(narr2, 2.5, 1.5).tolist() == [[1.5, 1.5, 12.0]]

    t = Tensor([1, 2, 3, 10, 0, 5, 10])
    assert map_value_tensor(t, 2, 10).tolist() == [1, 10, 3, 10, 0, 5, 10]

    narr3 = np.array([1, 2, 3])
    assert map_values_numpy_array(narr3, [1, 2, 3], [2, 3, 1]).tolist() == [2, 3, 1]


def test_gradient():
    tensor = Tensor([[1, 2, 6], [4, 8, 7]])
    grd_x = gradient(tensor, dim=1)
    grd_y = gradient(tensor, dim=0)
    assert grd_x.shape == tensor.shape == grd_y.shape
    assert grd_x.tolist() == [[1, 4, 4], [4, -1, -1]]
    assert grd_y.tolist() == [[3, 6, 1], [3, 6, 1]]
