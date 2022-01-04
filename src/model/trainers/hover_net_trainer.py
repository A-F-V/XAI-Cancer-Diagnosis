from src.model.metrics.hover_net_loss import HoVerNetLoss
from src.model.trainers.base_trainer import Base_Trainer

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.datasets.MoNuSeg import MoNuSeg
from src.transforms.image_processing.augmentation import *
import mlflow
import sys
import os
import io
import matplotlib.pyplot as plt
from PIL import Image
from src.vizualizations.cellseg_viz import generate_mask_diagram
from src.datasets.PanNuke import PanNuke
from src.model.architectures.graph_construction.hover_net import HoVerNet


class HoverNetTrainer(Base_Trainer):  # todo add one cycle learning
    def __init__(self, args):
        super(Base_Trainer, self).__init__()
        self.args = args

    def train(self):
        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")

        transforms = Compose([Normalize({"image": [0.6441, 0.4474, 0.6039]}, {"image": [
            0.1892, 0.1922, 0.1535]})])  # todo! correct

        device = args["DEVICE"]
        if device == "default":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        args["DEVICE"] = device

        print(f"Running on {device}")
        ds = PanNuke(transform=transforms) if args["DATASET"] == "PanNuke" else None
        dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=3)

        model = HoVerNet(self.args["RESNET_SIZE"])
        model.to(device)

        criterion = HoVerNetLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        # scheduler = None
        # if args["ONE_CYCLE"]:
        #    scheduler = optim.lr_scheduler.OneCycleLR(
        #        optimizer, max_lr=args['MAX_LR'], steps_per_epoch=1, epochs=args["EPOCHS"], #three_phase=False)

        print("Starting Training")
        with mlflow.start_run(args["RUN_ID"]):
            loop = tqdm(range(args["EPOCHS"]))
            mlflow.log_params(args)
            for epoch in loop:
                loss = train_step(model, dl, optimizer, criterion=criterion, args=args, loop=loop)
                loop.set_postfix_str(f"Loss: {loss}")
                mlflow.log_metric("training loss", loss, step=epoch+1)

                # if args["ONE_CYCLE"]:
                #    scheduler.step()
                #    mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch+1)

        #mlflow.pytorch.save_model(model, "trained_models/cell_seg_v1.pth")


def train_step(model, dataloader, optimizer, criterion, args, loop):
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
    print("Loop Started")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}")
        i, sm, hv = batch['image'], batch['semantic_mask'], batch['hover_map']

        x = i.to(args["DEVICE"])
        y1 = sm.to(args["DEVICE"])  # possibly use of epsilon to avoid log of zero
        y2 = hv.to(args["DEVICE"])

        y = (y1, y2)
        y_hat = model(x)

        loss = criterion(y_hat, y)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

    return train_loss/count
