from src.transforms.MoNuSeg import Normalize, ToTensor, RandomCrop
from random import random
from torch.nn import Conv2d
from torch import optim, nn
from torchvision.models.segmentation.fcn import FCNHead
from src.transforms.MoNuSeg import Normalize, ToTensor, RandomCrop
from src.vizualizations.tensor_viz import plot_tensor_histogram
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import sys
from pathlib import Path
from math import inf
from tqdm import tqdm

from torchvision.transforms import Compose, ToPILImage
import torch

from src.datasets.MoNuSeg import MoNuSeg
from torch.utils.data import DataLoader


class CellSegmentation_FCN(nn.Module):
    def __init__(self):
        super(CellSegmentation_FCN, self).__init__()

        def create_model():
            m = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
            m.classifier = nn.Sequential(FCNHead(2048, channels=1), nn.Sigmoid())
            return m

        self.model = create_model()

    def forward(self, image):
        return self.model(image)['out']

    def predict(self, image):
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.model.eval()
            image = image.to(device)
            output = self.model(image)['out']
            output = output.detach().cpu()
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            return output


def train_fn(model, dataloader, optimizer, criterion, args):
    """Performs one epoch's training.

    Args:
        model (nn.Module): The model being trained.
        dataloader (DataLoader): The DataLoader used for training.
        optimizer: The pytorch optimizer used
        criterion: The loss function
        args (dict): Additional arguments for training.
    """
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    train_loss = 0
    count = 0
    for i, batch in enumerate(dataloader):
        i, m = batch['image'], batch['semantic_mask']

        x = i.to(args["DEVICE"])
        y = m.to(args["DEVICE"])  # possibly use of epsilon to avoid log of zero

        y_hat = model(x)
        loss = criterion(y_hat, y)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

    return train_loss/count
