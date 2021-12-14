import numpy as np
import matplotlib.pyplot as plt
from .graph import Graph


def bounding_box_centre(img: np.ndarray):
    """
    Returns the bounding box of the image
    """
    cols = img.max(axis=0)
    rows = img.max(axis=1)
    y = (rows.nonzero()[0][0]+rows.nonzero()[0][-1])/2
    # because nonzero returns a tuple where the first element is the actually array
    x = (cols.nonzero()[0][0]+cols.nonzero()[0][-1])/2
    return x, y


def create_featureless_graph(inst_seg: np.ndarray):
    no_nuclei = inst_seg.max()
    graph = Graph()
    for nuceli_id in range(1, no_nuclei + 1):
        mask = inst_seg == nuceli_id
        centre = bounding_box_centre(mask)
        graph.add_node(centre)
    return graph
