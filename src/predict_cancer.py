import os
from src.model.architectures.graph_construction.hover_net import HoVerNet
from src.model.architectures.cancer_prediction.cell_unet_ae import UNET_AE
from src.model.architectures.cancer_prediction.pred_gnn import PredGNN
from PIL import Image
from torchvision.transforms import ToTensor
from src.transforms.graph_construction.hovernet_post_processing import instance_mask_prediction_hovernet
from src.transforms.graph_construction.graph_extractor import extract_graph, cell_to_voting_graph
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate, Distance
import torch
from torch_geometric.loader.dataloader import DataLoader
graph_trans = Compose([KNNGraph(6),  Distance(norm=False, cat=False)])

gnn_voter_args = {"LAYERS": 10, "WIDTH": 8, "GLOBAL_POOL": "MEAN", "RADIUS_FUNCTION": "INVSQUARE", "POOL_RATIO": 1}


def predict_cancer(img_loc, hover_net_loc=os.path.join("model", "HoVerNet.ckpt"), cell_predictor_loc=os.path.join("model", "CELL_PREDICTOR.ckpt"), gnn_voter_loc=os.path.join("model", "GNN_VOTER.ckpt")):
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

    with torch.no_grad():
        # Load the Models
        hover_net = HoVerNet.load_from_checkpoint(hover_net_loc).eval().cuda()
        cell_predictor = UNET_AE.load_from_checkpoint(cell_predictor_loc).eval().cuda()
        def cell_predict(X): return cell_predictor.forward_pred(X)
        gnn_voter = PredGNN.load_from_checkpoint(gnn_voter_loc, **gnn_voter_args).eval().cpu()

        # Load the image
        image = ToTensor()(Image.open(img_loc))

        # Create Instance Mask for cells
        instance_mask = instance_mask_prediction_hovernet(hover_net, image)

        del hover_net
        # Generate Cell Graph
        cell_graph = extract_graph(image, instance_mask)

        # Generate Voting Graph

        voting_graph = cell_to_voting_graph(cell_graph, cell_predict)
        voting_graph = graph_trans(voting_graph)

        del cell_predictor

        # Make Final Prediction

        prediction = gnn_voter(voting_graph.x, voting_graph.edge_index, voting_graph.edge_attr,
                               torch.zeros(voting_graph.x.shape[0]).long())

        del gnn_voter
        return prediction
