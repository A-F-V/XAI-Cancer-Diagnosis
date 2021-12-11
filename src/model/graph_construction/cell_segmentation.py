from torch.nn import Conv2d
from torch import optim, nn
from torchvision.models.segmentation.fcn import FCNHead
from src.transforms.MoNuSeg import Normalize, ToTensor
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


def train():
    top_folder = os.getcwd()

    transforms = Compose([ToTensor(), Normalize([0.6441, 0.4474, 0.6039], [0.1892, 0.1922, 0.1535])])

    def create_model():
        m = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)
        m.classifier = nn.Sequential(FCNHead(2048, channels=1), nn.Sigmoid())
        return m

    model = create_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_trans = DataLoader(MoNuSeg(os.path.join(top_folder, "data", "processed", "MoNuSeg"),
                                  transform=transforms), batch_size=1, shuffle=True, num_workers=1)

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    alpha = 0.95
    train_loss = 0.0

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        loop = tqdm(dl_trans, desc=f"Epoch {epoch} out of {epochs}")
        for i, batch in enumerate(loop):
            i, m = batch['image'], batch['semantic_mask']
            x = i.to(device)
            y = m.to(device)
            y_hat = model(x)['out']
            loss = criterion(y, y_hat)

            y = y.cpu()
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = alpha*train_loss + (1-alpha)*loss.item()
