import torch
from torch import Tensor, clip
from math import *
import math

x_axis = torch.tensor([1, 0, 0]).float()
y_axis = torch.tensor([0, 1, 0]).float()
z_axis = torch.tensor([0, 0, 1]).float()


def normalize_vec(vec: Tensor):
    vec = vec.float()
    return vec / vec.norm(p=2, dim=0)


def not_neg(vec: Tensor):
    return vec if vec[0] >= 0 else vec*-1


def rotate_in_plane(v: Tensor, u: Tensor, theta: float):  # todo make more precise
    """Rotate vector v around u by theta radians
    Args:
        v (Tensor): The vector to rotate
        u (Tensor): The axis of rotation
        theta (float): The angle of rotation clockwise (when facing the direction of u)
    Returns:
        v_new (Tensor): The rotated vector
    """
    u = normalize_vec(u.double())
    v_orig = v.clone()
    v = normalize_vec(v.double())

    alpha = torch.asin(u[0]).item()                 # the azimuth (between 0 and 360)
    if u[1].item() < 0:
        alpha = torch.pi-alpha
    beta = torch.acos(clip(u[2]/cos(alpha), -1, 1)).item()   # 90-elevation (between 0 and 180)
    #print("New Test")
    #print(u[1], cos(alpha), alpha)
    #print(alpha, beta)
    #print(v, u)
    #print(Tensor([sin(alpha), cos(alpha)*sin(beta), cos(alpha)*cos(beta)]))
    assert u.allclose(Tensor([sin(alpha), cos(alpha)*sin(beta), cos(alpha)*cos(beta)]), atol=1e-3)

    Rz = torch.tensor([[math.cos(alpha), math.sin(alpha), 0],
                       [-math.sin(alpha), math.cos(alpha), 0],
                       [0, 0, 1]])
    Rx = torch.tensor([[1, 0, 0],
                       [0, math.sin(beta), -math.cos(beta)],
                       [0, math.cos(beta), math.sin(beta)]])

    Ry = torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                       [0, 1, 0],
                       [-math.sin(theta), 0, math.cos(theta)]])

    v_prime = Rz.inverse() @ Rx.inverse() @ v
    # print(v_prime)
    v_prime = Ry @ v_prime
    # print(v_prime)
    v_final = Rx @ Rz @ v_prime
    # assert abs(find_angle(v_orig, u) - find_angle(v_final, u)) < 1e-2  # todo issue may be a lack of precision
    return v_final


def find_angle(v: Tensor, u: Tensor):
    """Calculate the angle between two vectors
    Args:
        v (Tensor): The first vector
        u (Tensor): The second vector
    Returns:
        The angle between the two vectors
    """
    v, u = normalize_vec(v), normalize_vec(u)
    dot = torch.dot(v, u)
    return torch.acos(dot)


def find_angle_of_rotation(v: Tensor, u: Tensor):
    """Calculates the angle of rotation (clockwise when looking in direction of perp) to rotate v to u along the axis that is perpendicular to both
    Args:
        v (Tensor): The first vector
        u (Tensor): The second vector
    Returns:
        The angle of rotation
    """
    v, u = normalize_vec(v), normalize_vec(u)
    normal = torch.cross(v, u)
    mag = normal.norm(p=2)
    theta = torch.asin(mag)
    return theta
