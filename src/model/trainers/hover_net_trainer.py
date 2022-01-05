
from src.model.metrics.hover_net_loss import HoVerNetLoss
from src.model.trainers.base_trainer import Base_Trainer

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply
from src.datasets.MoNuSeg import MoNuSeg
from src.transforms.image_processing.augmentation import *
import mlflow
import matplotlib.pyplot as plt
import io
from PIL import Image
from src.vizualizations.cellseg_viz import generate_mask_diagram
from src.datasets.PanNuke import PanNuke
from src.model.architectures.graph_construction.hover_net import HoVerNet
from src.vizualizations.image_viz import plot_images
from src.utilities.img_utilities import tensor_to_numpy
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class HoverNetTrainer(Base_Trainer):
    def __init__(self, args):
        super(Base_Trainer, self).__init__()
        self.args = args

    def train(self):
        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")
        print("Getting the Data")

        transforms = Compose([
            Normalize(
                {"image": [0.6441, 0.4474, 0.6039]},
                {"image": [0.1892, 0.1922, 0.1535]}),
            RandomCrop((200, 200)),  # does not work in random apply as will cause batch to have different sized pictures
            RandomApply(
                [
                    # RandomRotate(), - not working #todo! fix
                    RandomFlip()
                    #AddGaussianNoise(0.01, fields=["image"]),
                    #ColourJitter(bcsh=(0.1, 0.1, 0.1, 0.1), fields=["image"])
                ],
                p=0.5)
        ])  # consider adding scale
        dataset = PanNuke(transform=transforms)
        train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"])
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE"], shuffle=False, num_workers=args["NUM_WORKERS"])

        num_training_batches = len(train_loader)*args["EPOCHS"]
        model = HoVerNet(num_training_batches, train_loader=train_loader, val_loader=val_loader, ** args)

        mlf_logger = MLFlowLogger(experiment_name="HoVerNet", run_name=args["RUN_NAME"])
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(gpus=1, max_epochs=args["EPOCHS"], logger=mlf_logger, callbacks=[
                             lr_monitor, EarlyStopping(monitor="val_loss")])

        print("Training Starting")

        with mlflow.start_run(experiment_id=2, run_name=args["RUN_NAME"]) as run:
            lr_finder = trainer.tuner.lr_find(model, num_training=1000)
            fig = lr_finder.plot(suggest=True)
            fig.show()

            trainer.fit(model)
