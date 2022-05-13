from locale import normalize
import os
from src.transforms.image_processing.he_normalize import normalize_he_image
from src.deep_learning.architectures.graph_construction.hover_net import HoVerNet
from src.deep_learning.architectures.cancer_prediction.cell_encoder import CellEncoder
from src.deep_learning.architectures.cancer_prediction.cancer_gnn import CancerGNN
from PIL import Image
from torchvision.transforms import ToTensor
from src.transforms.cell_segmentation.hovernet_post_processing import instance_mask_prediction_hovernet, cut_img_from_tile
from src.transforms.graph_construction.node_embedding import generate_node_embeddings
from src.transforms.graph_construction.graph_extractor import extract_graph, cell_to_voting_graph
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate, Distance
import torch
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.backends.backend_pdf
from src.utilities.img_utilities import tensor_to_numpy
import matplotlib.pyplot as plt
from src.transforms.cell_segmentation.percolation import hollow
from numpy.ma import masked_where
import numpy as np
import torch
from torch import Tensor
from src.transforms.image_processing.filters import glcm

graph_trans = Compose([KNNGraph(6),  Distance(norm=False, cat=False)])

gnn_args = {"HEIGHT": 7, "WIDTH": 32}


# ADD NORMALIZER

def predict_cancer(img_loc, hover_net_loc=os.path.join("model", "HoVerNet.ckpt"), resnet_encoder=os.path.join("model", "CellEncoder.ckpt"), gnn_loc=os.path.join("model", "GCN.ckpt"), explainability_location=None, concept_folder=None):
    exp = explainability_location != None
    # if exp:
    #    file_name = os.path.basename(img_loc).split(".")[0]
    #    pdf_path = os.path.join(explainability_location, f"{file_name}.pdf")
    #    if os.path.exists(pdf_path):
    #        os.remove(pdf_path)
    #    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    with torch.no_grad():
        # Load the Models
        hover_net = HoVerNet.load_from_checkpoint(hover_net_loc).eval().cuda()
        resnet_encoder = CellEncoder.load_from_checkpoint(
            os.path.join("model", "CellEncoder.ckpt")).eval().cuda()
        gnn = CancerGNN.load_from_checkpoint(gnn_loc, **gnn_args).eval().cuda()

        # Load the image
        image = ToTensor()(Image.open(img_loc))

        # 1) Stain Normalisation

        normalized_image = normalize_he_image(image)

        # 2) Cell Segmentation

        # Create Instance Mask for cells
        instance_mask, category_mask = instance_mask_prediction_hovernet(hover_net, normalized_image)
        normalized_image_cropped = cut_img_from_tile(image, 128)

        del hover_net

        # 3) Graph Construction
        # Generate Cell Graph

        crop_graph = extract_graph(normalized_image_cropped, instance_mask, category_mask).cuda()
        glcm_for_graph = torch.zeros((crop_graph.x.shape[0], 50))

        for i, img in enumerate(crop_graph.x):
            glcm_for_graph[i] = glcm(img, normalize=True)
        cell_graph = crop_graph.clone()
        cell_graph.x = generate_node_embeddings(imgs=crop_graph.x, resnet_encoder=resnet_encoder,
                                                num_neighbours=crop_graph.x, cell_types=crop_graph.cell_types,
                                                glcm=glcm_for_graph)

        del cell_predictor

        # 4) Graph Classification

        prediction = gnn.predict(cell_graph).long().cuda().squeeze()

        # 5) GCExplainability

        # if exp:
        #    save_original_image(pdf, image_cropped)
        #    save_instance_mask_vizualization(pdf, image_cropped, instance_mask)
        #    save_prediction_certainity_bar_chart(pdf, prediction)
        #    pdf.close()
        return prediction


# def save_original_image(pdf, image):
#    fig = plt.figure(figsize=(20, 20))
#    plt.imshow(tensor_to_numpy(image))
#    plt.axis("off")
#    plt.title("Original Image")
#    pdf.savefig(fig)
#
#
# def save_instance_mask_vizualization(pdf, orig_img, inst_mask):
#    instance_mask_border = hollow(inst_mask)
#
#    fig = plt.figure(figsize=(20, 20))
#    plt.imshow(tensor_to_numpy(orig_img))
#  # todo, better border
#    plt.imshow(masked_where(instance_mask_border != 0, instance_mask_border),
#               cmap="nipy_spectral", alpha=0.7)
#    plt.axis("off")
#    plt.title("Identified Cells")
#    pdf.savefig(fig)
#
#
# def save_prediction_certainity_bar_chart(pdf, prediction: Tensor):
#    types = ["Normal", "Benign", "In Situ", "Invasive"]
#    x_pos = np.arange(len(types))
#
#    fig = plt.figure(figsize=(20, 20))
#
#    plt.bar(x=x_pos, height=[prediction[3],
#            prediction[0], prediction[1], prediction[2]], align='center')
#    plt.xticks(x_pos, types)
#    plt.title("Model Certainty")
#    pdf.savefig(fig)
#
#    # EXPLAINABILITY VIZUALIZATIONS
#    # - [x] GNN probability predictions bar chart
#    # - [x] Instance Mask over image
#    # - [ ] Cell Graph over image
#    # - [ ] Simple Gradient Based projection on image
#    # - [ ] GCN Explainer
#    # - [ ] Refactor code to an explainability folder
