from src.utilities.vector_utilities import *


def test_normalize_vec():
    v = torch.tensor([1, 2, 3]).float()
    v_norm = normalize_vec(v)
    assert abs(v_norm.norm(p=2, dim=0)-1) < 1e-5


def test_rotate_in_plane_specific_cases():
    pi = torch.pi
    assert rotate_in_plane(x_axis, y_axis, pi/2).allclose(-z_axis, atol=1e-5)
    assert rotate_in_plane(x_axis, z_axis, pi/2).allclose(y_axis, atol=1e-5)
    assert rotate_in_plane(x_axis, torch.Tensor([1, 1, 0]).float(), pi).allclose(y_axis, atol=1e-3)
    assert rotate_in_plane(y_axis, torch.Tensor([1, -1, 0]).float(), pi).allclose(-x_axis, atol=1e-3)
    assert rotate_in_plane(z_axis, -y_axis, 3*pi/2).allclose(x_axis, atol=1e-3)
    assert rotate_in_plane(x_axis, torch.Tensor([1, 1, 1]), 2*pi/3).allclose(y_axis, atol=1e-3)

    # todo test more precise version


def test_rotate_in_plane_variety_of_cases():
    grad = 20
    inc = torch.pi*2/grad
    for angle in range(0, grad):
        angle = angle * inc
        vec = Tensor([cos(angle), sin(angle), 0]).float()
        ([rotate_in_plane(y_axis*2+z_axis, vec, theta*inc) for theta in range(0, grad)])


def test_find_angle():
    pi = torch.pi
    assert find_angle(x_axis, y_axis) == pi/2


def test_find_angle_of_rotation():
    pi = torch.pi
    assert find_angle(y_axis, x_axis) == pi/2
