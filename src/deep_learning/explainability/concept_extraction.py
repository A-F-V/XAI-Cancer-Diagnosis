import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def disect_concept_graph(concept_graph, min_subgraph_size=5):
    graph = concept_graph.clone()

    left_nodes, right_nodes = concept_graph.x[graph.edge_index[0, :]].argmax(
        dim=1), concept_graph.x[graph.edge_index[1, :]].argmax(dim=1)

    keep_edge = left_nodes == right_nodes
    graph.edge_index = graph.edge_index[:, keep_edge]

    G = to_networkx(graph, to_undirected=True, node_attrs=['x', 'activation', 'pos'])

    sub_graphs = [G.subgraph(g) for g in nx.components.connected_components(G) if len(g) >= min_subgraph_size]
    for i, g in enumerate(sub_graphs):
        g = from_networkx(g, group_node_attrs=['activation', 'x', 'pos'])
        g.pos = g.x[:, -2:]
        g.activation = g.x[:, :32]
        g.x = g.x[:, 32:-2]
        g.graph_id = graph.graph_id
        g.y = graph.y.item()

        g.concept = g.x[0].argmax()
        sub_graphs[i] = g
    return sub_graphs


def load_concept_information(concept_folder):
    files = os.listdir(concept_folder)
    concept_means = np.load(os.path.join(concept_folder, "concept_means.npy"))
    k = len(concept_means)
    exemplary_images = [[] for i in range(k)]
    for concept in range(k):
        for example in range(3):
            file = os.path.join(concept_folder, f"c{concept}e{example}.png")
            if file in files:
                image = Image.open(file)
                exemplary_images[concept].append(image)
    class_concept_prob = np.load(os.path.join(concept_folder, "class_concept_prob.npy"))
    mu = np.load(os.path.join(concept_folder, "mu.npy"))
    sigma = np.load(os.path.join(concept_folder, "sigma.npy"))
    return concept_means, exemplary_images, class_concept_prob, mu, sigma
