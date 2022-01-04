from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.datasets.MoNuSeg import MoNuSeg
from src.transforms.pytorch.MoNuSeg import *
from src.model.architectures.graph_construction.fcn import CellSegmentation_FCN, train_fn
import mlflow
import sys
import os
import io
import matplotlib.pyplot as plt
from PIL import Image
from src.vizualizations.cellseg_viz import generate_mask_diagram


def run():
    top_folder = os.getcwd()
    print(top_folder)

    transforms = Compose([ToTensor(), Normalize([0.6441, 0.4474, 0.6039], [
        0.1892, 0.1922, 0.1535]), RandomCrop(size=(250, 250))])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dl = DataLoader(MoNuSeg(transform=transforms), batch_size=16, shuffle=True, num_workers=3)

    args = {"DEVICE": device, "EPOCHS": 100, "ONE_CYCLE": True, "MAX_LR": 0.005}

    model = CellSegmentation_FCN()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = None
    if args["ONE_CYCLE"]:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args['MAX_LR'], steps_per_epoch=1, epochs=args["EPOCHS"], three_phase=False)

    with mlflow.start_run():
        loop = tqdm(range(args["EPOCHS"]))
        mlflow.log_params(args)
        for epoch in loop:
            loss = train_fn(model, dl, optimizer, criterion=criterion, args=args)
            loop.set_postfix_str(f"Loss: {loss}")
            mlflow.log_metric("training loss", loss, step=epoch+1)

            if args["ONE_CYCLE"]:
                scheduler.step()
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch+1)

        img_buf = io.BytesIO()
        f = generate_mask_diagram(model, dl, args=args)
        f.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        mlflow.log_image(im, "mask_diagram.png")
       #mlflow.pytorch.save_model(model, "trained_models/cell_seg_v1.pth")
