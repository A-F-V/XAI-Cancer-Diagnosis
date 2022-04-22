import os
from src.model.architectures.graph_construction.hover_net import HoVerNet
from src.model.architectures.cancer_prediction.cell_encoder import CellEncoder
from src.model.architectures.cancer_prediction.pred_gnn import PredGNN
from PIL import Image
from torchvision.transforms import ToTensor
from src.transforms.graph_construction.hovernet_post_processing import instance_mask_prediction_hovernet, cut_img_from_tile
from src.transforms.graph_construction.graph_extractor import extract_graph, cell_to_voting_graph
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate, Distance
import torch
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.backends.backend_pdf
from src.utilities.img_utilities import tensor_to_numpy
import matplotlib.pyplot as plt
from src.transforms.graph_construction.percolation import hollow
from numpy.ma import masked_where
import numpy as np
import torch
from torch import Tensor
graph_trans = Compose([KNNGraph(6),  Distance(norm=False, cat=False)])

gnn_voter_args = {"LAYERS": 12, "WIDTH": 4, "GLOBAL_POOL": "MEAN", "RADIUS_FUNCTION": "INVSQUARE", "POOL_RATIO": 1}


def predict_cancer(img_loc, hover_net_loc=os.path.join("model", "HoVerNet.ckpt"), cell_predictor_loc=os.path.join("model", "CELL_PREDICTOR.ckpt"), gnn_voter_loc=os.path.join("model", "GNN_VOTER.ckpt"), explainability_location=None):
    """
    Predict cancer from image.

    Args:
        img_loc (str): Location of image to predict.
        hover_net_loc (str): Location of hover net model.
        cell_predictor_loc (str): Location of cell predictor model.
        gnn_voter_loc (str): Location of gnn voter model.

    Returns:
        str: Cancer type.
    """
    # todo check if fewer than 10 cells are identified
    exp = explainability_location != None
    if exp:
        file_name = os.path.basename(img_loc).split(".")[0]
        pdf_path = os.path.join(explainability_location, f"{file_name}.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    with torch.no_grad():
        # Load the Models
        hover_net = HoVerNet.load_from_checkpoint(hover_net_loc).eval().cuda()
        cell_predictor = CellEncoder.load_from_checkpoint(cell_predictor_loc).eval().cuda()
        def cell_predict(X): return cell_predictor.forward_pred(X)
        gnn_voter = PredGNN.load_from_checkpoint(gnn_voter_loc, **gnn_voter_args).eval().cuda()

        # Load the image
        image = ToTensor()(Image.open(img_loc))

        # Create Instance Mask for cells
        instance_mask, category_mask = instance_mask_prediction_hovernet(hover_net, image)
        image_cropped = cut_img_from_tile(image, 128)

        del hover_net
        # Generate Cell Graph
        cell_graph = extract_graph(image_cropped, instance_mask).cuda()

        # Generate Voting Graph

        voting_graph = cell_to_voting_graph(cell_graph, cell_predict)
        voting_graph = graph_trans(voting_graph).cuda()

        del cell_predictor

        # Make Final Prediction

        prediction = gnn_voter(voting_graph.x, voting_graph.edge_index, voting_graph.edge_attr,
                               torch.zeros(voting_graph.x.shape[0]).long().cuda()).squeeze()

        del gnn_voter

        if exp:
            save_original_image(pdf, image_cropped)
            save_instance_mask_vizualization(pdf, image_cropped, instance_mask)
            save_prediction_certainity_bar_chart(pdf, prediction)
            pdf.close()
        return prediction


def save_original_image(pdf, image):
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(tensor_to_numpy(image))
    plt.axis("off")
    plt.title("Original Image")
    pdf.savefig(fig)


def save_instance_mask_vizualization(pdf, orig_img, inst_mask):
    instance_mask_border = hollow(inst_mask)

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(tensor_to_numpy(orig_img))
  # todo, better border
    plt.imshow(masked_where(instance_mask_border != 0, instance_mask_border),
               cmap="nipy_spectral", alpha=0.7)
    plt.axis("off")
    plt.title("Identified Cells")
    pdf.savefig(fig)


def save_prediction_certainity_bar_chart(pdf, prediction: Tensor):
    types = ["Normal", "Benign", "In Situ", "Invasive"]
    x_pos = np.arange(len(types))

    fig = plt.figure(figsize=(20, 20))

    plt.bar(x=x_pos, height=[prediction[3],
            prediction[0], prediction[1], prediction[2]], align='center')
    plt.xticks(x_pos, types)
    plt.title("Model Certainty")
    pdf.savefig(fig)

    # EXPLAINABILITY VIZUALIZATIONS
    # - [x] GNN probability predictions bar chart
    # - [x] Instance Mask over image
    # - [ ] Cell Graph over image
    # - [ ] Simple Gradient Based projection on image
    # - [ ] GCN Explainer
    # - [ ] Refactor code to an explainability folder
