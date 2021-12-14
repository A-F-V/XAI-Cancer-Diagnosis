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


def create_featureless_graph(inst_seg: np.ndarray, dist_threshold: int = 30):
    no_nuclei = inst_seg.max()
    graph = Graph()
    for nuceli_id in range(1, no_nuclei + 1):
        mask = inst_seg == nuceli_id
        centre = bounding_box_centre(mask)
        graph.add_node({'centre': centre})
    connect_nuclei(graph, dist_threshold)
    return graph


def euclidean_distance(p1, p2):
    p1n, p2n = np.array(p1), np.array(p2)
    return np.linalg.norm(p1n-p2n)


def connect_nuclei(graph: Graph, dist_threshold: int = 30):
    node_count = len(graph.nodes)
    for node in range(node_count):
        for other_node in range(node+1, node_count):
            dist = euclidean_distance(graph.nodes[node]['centre'], graph.nodes[other_node]['centre'])
            if dist < dist_threshold:
                graph.add_edge(node, other_node, 1)
