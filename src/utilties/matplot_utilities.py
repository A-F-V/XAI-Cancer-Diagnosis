import torch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


from src.utilties.vector_utilities import normalize_vec


def draw_plane(ax, v1, v2, color='r', alpha=0.2):
    """
    Draws a plane in the given axis through the origin
    """
    origin = torch.zeros(3)
    # get the plane points
    normal = normalize_vec(torch.cross(v1, v2))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 100))
    z = (-normal[0]*xx - normal[1]*yy)/normal[2]
    # draw the plane
    ax.plot_surface(xx, yy, z,
                    color=color, alpha=alpha)


def draw_vectors_3d(ax, vectors, length=1, color='r'):
    ax.quiver(np.zeros(len(vectors)), np.zeros(len(vectors)), np.zeros(
        len(vectors)), vectors[:, 0], vectors[:, 1], vectors[:, 2], length=length, color=color)


def draw_vector_3d(ax, vector, centre=(0, 0, 0), length=1, color='r'):
    ax.quiver(centre[0], centre[1], centre[2], vector[0], vector[1], vector[2], length=length, color=color)


def draw_annotated_vector_3d(ax, vector, centre, text, length=1, color='r'):  # todo fix this
    draw_vector_3d(ax, vector, centre, length=length, color=color)
    ax.text3D(centre[0], centre[1], centre[2], text, zdir=vector)


def draw_vector_2d(ax, vector, centre=(0, 0), color='r'):
    ax.arrow(centre[0], centre[1], vector[0], vector[1], color=color, shape="right")
