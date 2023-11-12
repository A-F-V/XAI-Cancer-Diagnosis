from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from src.utilities.img_utilities import tensor_to_numpy
from torchvision.transforms import ToTensor
from src.transforms.cell_segmentation.hovernet_post_processing import cut_img_from_tile
from scipy.spatial import ConvexHull
from PIL import Image
from src.deep_learning.explainability.concept_extraction import disect_concept_graph
import torch
import numpy as np


def create_point_sphere(centre: torch.Tensor, radius, n_points):
    output = []
    for theta in np.linspace(0, 2*np.pi, n_points):

        output += [centre + radius *
                   torch.as_tensor([torch.cos(torch.as_tensor(theta)), torch.sin(torch.as_tensor(theta))])]
    return torch.stack(output)


def visualise_concept_subgraphs(sgs, image_loc, concept_means, save=False, save_loc=None, ax=None, crop=False):
    if ax is None:
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
    plt.axis('off')
    # Load the image
    img = tensor_to_numpy(cut_img_from_tile(ToTensor()(Image.open(image_loc)), tile_size=128))
    # Place nodes on the image
    # Place smoothing sphere around cells
    # Convex hull around
    for sg in sgs:
        points = torch.zeros(0, 2)
        centres = []
        distances = ((sg.activation - concept_means[sg.concept])**2).sum(dim=1)
        height = []
        for i in range(len(sg.x)):
            centre = sg.pos[i]
            centres.append(centre)
            height.append(distances[i])
            cell_points = create_point_sphere(centre, 64, 10)
            points = torch.cat([points, cell_points])
        points = points[ConvexHull(points.numpy()).vertices]
        points = torch.cat([points, points[0].unsqueeze(0)])
        ax.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'b-', lw=5)

    if(crop):
        padding = 10
        xmin = int(max((points[:, 0].numpy()-padding).min(), 0))
        xmax = int(min((points[:, 0].numpy()+padding).max(), img.shape[1]))
        ymin = int(max((points[:, 1].numpy()-padding).min(), 0))
        y_max = int(min((points[:, 1].numpy()+padding).max(), img.shape[0]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, y_max)

    ax.imshow(img, aspect='auto')

    if(save and save_loc is not None):
        plt.savefig(save_loc)
        plt.close(f)


def find_primary_evidence(concept_sub_graphs, prediction, p_class_given_concept):
    evidence = sorted(list(filter(lambda g: p_class_given_concept[g.concept][prediction] > 50, concept_sub_graphs)),
                      key=lambda g: p_class_given_concept[g.concept][prediction], reverse=True)
    primary_concept = evidence[0].concept
    primary_evidence = list(filter(lambda g: g.concept == primary_concept, evidence))
    return primary_evidence


def explain_prediction(concept_graph, img_loc, prediction, means, k, p_class_given_concept, exemplary_concept_graphs_final, save_loc=None):

    sgs = disect_concept_graph(concept_graph, min_subgraph_size=5)
    assert p_class_given_concept.shape == (k, 4)

    primary_evidence = find_primary_evidence(sgs, prediction, p_class_given_concept)
    primary_concept = primary_evidence[0].concept.item()

    f = plt.figure(figsize=(20, 10))
    gs = GridSpec(nrows=3, ncols=7)
    ax_main = f.add_subplot(gs[:, :5])
    ax_main.title.set_text('Prediction:\n'+['Normal', 'Benign', 'In-Situ', 'Invasive'][prediction])
    ax_main.title.set_fontsize(40)
    visualise_concept_subgraphs(primary_evidence, img_loc, means, ax=ax_main)
    ax_concepts = [f.add_subplot(gs[i:i+1, 6:]) for i in range(3)]
    ax_concepts[0].title.set_text('Supporting Concept:\n#'+str(primary_concept))
    ax_concepts[0].title.set_fontsize(40)
    for i, concept_img in enumerate(exemplary_concept_graphs_final[primary_concept]):
        ax_concepts[i].axis('off')
        ax_concepts[i].imshow(concept_img, aspect='auto')
    if save_loc is not None:
        plt.savefig(save_loc)
        plt.close(f)
