
from gc import callbacks
from src.model.metrics.hover_net_loss import HoVerNetLoss
from src.model.trainers.base_trainer import Base_Trainer
import os
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply, RandomChoice
from src.datasets.MoNuSeg import MoNuSeg
from src.transforms.image_processing.augmentation import *
import mlflow
import matplotlib.pyplot as plt
import io
from PIL import Image
from src.vizualizations.cellseg_viz import cell_segmentation_sliding_window_gif_example, generate_mask_diagram
from src.datasets.PanNuke import PanNuke
from src.model.architectures.graph_construction.hover_net import HoVerNet
from src.vizualizations.image_viz import plot_images
from src.utilities.img_utilities import tensor_to_numpy
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utilities.mlflow_utilities import log_plot
import numpy as np
from src.datasets.train_val_split import train_val_split
from src.transforms.graph_construction.hovernet_post_processing import instance_mask_prediction_hovernet
from src.model.metrics.panoptic_quality import panoptic_quality

# todo
"""
Instead of using random_split, you could create two datasets, one training dataset with the random transformations, and another validation set with its corresponding transformations.
Once you have created both datasets, you could randomly split the data indices e.g. using sklearn.model_selection.train_test_split. These indices can then be passed to torch.utils.data.Subset together with their datasets in order to create the final training and validation dataset.
"""
scale_modes = {"image": InterpolationMode.BILINEAR,
               "semantic_mask": InterpolationMode.NEAREST, "instance_map": InterpolationMode.NEAREST}
transforms_training = Compose([
    Compose([
        RandomChoice([
            RandomScale(x_fact_range=(0.45, 0.55), y_fact_range=(0.45, 0.55),
                        modes=scale_modes),
            RandomScale(x_fact_range=(0.65, 0.75), y_fact_range=(0.65, 0.75),
                        modes=scale_modes),
            RandomScale(x_fact_range=(0.95, 1.05), y_fact_range=(0.95, 1.05),
                        modes=scale_modes),

        ], p=(0.05, 0.05, 0.9)),
        RandomCrop(size=(256, 256))  # 64 for PanNuke,128 for MoNuSeg
    ]),
    # RandomCrop((64, 64)),  # does not work in random apply as will cause batch to have different sized pictures

    RandomApply(
        [
            # RandomRotate(), - not working #todo! fix
            RandomFlip(),
            AddGaussianNoise(0.01, fields=["image"]),
            ColourJitter(bcsh=(0.2, 0.1, 0.1, 0.01), fields=["image"]),
            GaussianBlur(fields=["image"])
        ],

        p=0.5),
    Normalize(
        {"image": [0.6441, 0.4474, 0.6039]},
        {"image": [0.1892, 0.1922, 0.1535]})
])

transforms_val = Compose([
    RandomCrop(size=(128, 128)),
    Normalize(
        {"image": [0.6441, 0.4474, 0.6039]},
        {"image": [0.1892, 0.1922, 0.1535]})
])


class HoverNetTrainer(Base_Trainer):
    def __init__(self, args):
        super(Base_Trainer, self).__init__()
        self.args = args

    def train(self):
        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")
        print("Getting the Data")

        # consider adding scale
        if args["DATASET"] == "MoNuSeg":
            src_folder = os.path.join("data", "processed",
                                      "MoNuSeg_TRAIN")
            train_set, val_set = train_val_split(MoNuSeg, src_folder, 0.8, transforms_training, transforms_val)

        elif args["DATASET"] == "PanNuke":
            src_folder = os.path.join("data", "processed",
                                      "PanNuke")
            train_set, val_set = train_val_split(PanNuke, src_folder, 0.8, transforms_training, transforms_val)
            #dataset = PanNuke(transform=transforms_training)
            # train_set, val_set = random_split(
            #    dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                                shuffle=False, num_workers=args["NUM_WORKERS"], persistent_workers=True)

        num_training_batches = len(train_loader)*args["EPOCHS"]

        model = None
        if args["START_CHECKPOINT"]:
            print(f"Model is being loaded from checkpoint {args['START_CHECKPOINT']}")
            checkpoint_path = make_checkpoint_path(args["START_CHECKPOINT"])
            model = HoVerNet.load_from_checkpoint(
                checkpoint_path, num_batches=num_training_batches, train_loader=train_loader, val_loader=val_loader, ** args)

            # model.encoder.freeze()
        else:
            model = HoVerNet(num_training_batches, train_loader=train_loader, val_loader=val_loader, ** args)
        mlf_logger = MLFlowLogger(experiment_name=args["EXPERIMENT_NAME"], run_name=args["RUN_NAME"])

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer_callbacks = [
            lr_monitor
        ]
        if args["EARLY_STOP"]:
            trainer_callbacks.append(EarlyStopping(monitor="val_loss"))

        trainer = pl.Trainer(log_every_n_steps=1, gpus=1,
                             max_epochs=args["EPOCHS"], logger=mlf_logger, callbacks=trainer_callbacks,
                             enable_checkpointing=True, default_root_dir=os.path.join("experiments", "checkpoints"),
                             profiler="simple",)

        ###########
        # EXTRAS  #
        ###########
        model.encoder.unfreeze()

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:
                lr_finder = trainer.tuner.lr_find(model, num_training=100, max_lr=10)
                fig = lr_finder.plot(suggest=True)
                log_plot(fig, "LR_Finder")
                print(lr_finder.suggestion())
        else:
            print("Training Started")
            trainer.fit(model)
            #print("Training Over\nEvaluating")
            # trainer.validate(model)
            ckpt_file = str(args['EXPERIMENT_NAME'])+"_"+str(args['RUN_NAME'])+".ckpt"
            ckpt_path = make_checkpoint_path(ckpt_file)
            trainer.save_checkpoint(ckpt_path)

    def run(self, checkpoint):
        args = self.args
        model = HoVerNet.load_from_checkpoint(checkpoint, **args)
        model.eval()
        model.cpu()
        dataset = MoNuSeg(src_folder=os.path.join("data", "processed",
                                                  "MoNuSeg_TRAIN"), transform=transforms_val)
        imgs = []
        with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=f"DIAG_{os.path.basename(checkpoint)}") as run:
            for i in range(10):
                sample = dataset[i]
                img = sample["image"].unsqueeze(0)
                sm = sample["semantic_mask"]
                imgs.append(sm.squeeze().detach())
                sm_hat, _ = model(img)
                imgs.append(sm_hat.squeeze().detach())
                imgs.append((sm_hat > 0.5).squeeze().detach())
            plot_images(imgs, (10, 3), cmap="gray")
            log_plot(plt, "Prediction Diagnosis")

            cell_segmentation_sliding_window_gif_example(
                model, dataset[0], location=os.path.join("experiments", "artifacts", "cell_seg_img.gif"))
            mlflow.log_artifact(os.path.join("experiments", "artifacts", "cell_seg_img.gif"))

            # CALCULATE MEAN PANOPTIC QUALITY
            dataset = MoNuSeg(src_folder=os.path.join("data", "processed",
                                                      "MoNuSeg_TRAIN"), transform=Compose([]))
            pq_tot = 0
            for i in range(10):
                sample = dataset[i]
                ins_pred = instance_mask_prediction_hovernet(model, sample['image'], tile_size=128)
                pq = panoptic_quality(ins_pred.squeeze(), sample['instance_mask'].squeeze()[64:768+64, 64:768+64])
                pq_tot += pq
            pq_tot /= 10
            mlflow.log_metric("Mean Panoptic Quality", pq_tot)

        pass


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)
