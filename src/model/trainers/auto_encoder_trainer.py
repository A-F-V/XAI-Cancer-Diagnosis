
from src.model.trainers.base_trainer import Base_Trainer
import os
from tqdm import tqdm
from torch_geometric.loader.dataloader import DataLoader
from src.transforms.image_processing.augmentation import *
import mlflow
from src.datasets.BACH import BACH
import pytorch_lightning as pl
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utilities.mlflow_utilities import log_plot
import numpy as np
from src.model.architectures.cancer_prediction.cancer_net import CancerNet
from src.model.architectures.cancer_prediction.simple_gnn import SimpleGNN
import json
from src.model.architectures.cancer_prediction.cell_autoencoder import CellAutoEncoder
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate
from src.datasets.BACH_Cells import BACH_Cells
from src.transforms.graph_augmentation.edge_dropout import EdgeDropout, far_mass
from src.datasets.train_val_split import train_val_split
from src.transforms.image_processing.augmentation import *


class CellAETrainer(Base_Trainer):
    def __init__(self, args=None):
        super(Base_Trainer, self).__init__()
        if args == None:
            args = json.load(open(os.path.join("experiments", "args", "default.json")))
        self.args = args

    def train(self):

        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")
        print("Getting the Data")

        tr_trans = Compose([
            RandomFlip(fields=["img"]), AddGaussianNoise(0.1, fields=['img'])]
        )  # ! TODO
        val_trans = Compose([])

        src_folder = os.path.join("data", "processed",
                                  "BACH_TRAIN")

        # BACH_Cells(src_folder).compile_cells()
        train_set, val_set = train_val_split(BACH_Cells, src_folder, 0.8, tr_trans=tr_trans, val_trans=val_trans)

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                                shuffle=False, num_workers=args["NUM_WORKERS"], persistent_workers=True)

        accum_batch = max(1, 64//args["BATCH_SIZE_TRAIN"])
        num_steps = len(train_loader)*args["EPOCHS"]//accum_batch

        print(f"Using {len(train_set)} training examples and {len(val_set)} validation example - With #{num_steps} steps")

        model = CellAutoEncoder(img_size=args["IMG_SIZE"], num_steps=num_steps,
                                val_loader=val_loader, train_loader=train_loader, layers=args["TISSUE_RADIUS"], **args)

        # if args["START_CHECKPOINT"]:
        #    print(f"Model is being loaded from checkpoint {args['START_CHECKPOINT']}")
        #    checkpoint_path = make_checkpoint_path(args["START_CHECKPOINT"])
        #    model = CancerNet.load_from_checkpoint(
        #        checkpoint_path, num_batches=num_training_batches, train_loader=train_loader, val_loader=val_loader, degree_dist=node_dist,
        #        down_samples=args["DOWN_SAMPLES"],
        #        img_size=args["IMG_SIZE"],
        #        tissue_radius=args["TISSUE_RADIUS"],** args)
        #    # model.encoder.freeze()
        # else:
        #    model = CancerNet(degree_dist=node_dist, num_batches=num_training_batches,
        #                      train_loader=train_loader, val_loader=val_loader,
        #                      down_samples=args["DOWN_SAMPLES"],
        #                      img_size=args["IMG_SIZE"],
        #                      tissue_radius=args["TISSUE_RADIUS"], ** args)

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
                             profiler="simple",
                             accumulate_grad_batches=accum_batch,)

        ###########
        # EXTRAS  #
        ###########

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:
                lr_finder = trainer.tuner.lr_find(model, num_training=1000, max_lr=0.01)
                fig = lr_finder.plot(suggest=True)
                log_plot(fig, "LR_Finder")
                print(lr_finder.suggestion())
        else:
            print("Training Started")
            trainer.fit(model)
            # print("Training Over\nEvaluating")
            # trainer.validate(model)
            ckpt_file = str(args['EXPERIMENT_NAME'])+"_"+str(args['RUN_NAME'])+".ckpt"
            ckpt_path = make_checkpoint_path(ckpt_file)
            trainer.save_checkpoint(ckpt_path)

    def run(self, checkpoint):
        pass


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)
