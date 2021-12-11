from torch.nn import Conv2d
from torch import optim, nn
from torchvision.models.segmentation.fcn import FCNHead
from src.transforms.MoNuSeg import Normalize, ToTensor

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
        m.classifier = FCNHead(2048, channels=1)
        return m

    model = create_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_trans = DataLoader(MoNuSeg(os.path.join(top_folder, "data", "processed", "MoNuSeg"),
                                  transform=transforms), batch_size=1, shuffle=True, num_workers=1)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    alpha = 0.95
    train_loss = 0.0

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(dl_trans), desc=f"Trainining Loss: {train_loss}. Epoch {epoch} out of {epochs}"):
            i, m = batch['image'], batch['semantic_mask']
            x = i.to(device).float()
            y = m.to(device).float()
            y_hat = model(x)['out']
            print(y.shape)
            print(y_hat.shape)
            loss = criterion(y, y_hat)
            del x
            del y
            del y_hat
            torch.cuda.empty_cache()
            break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = alpha*train_loss + (1-alpha)*loss.item()
