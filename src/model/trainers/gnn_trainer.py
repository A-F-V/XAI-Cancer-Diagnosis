
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
import torch

node_dist = torch.as_tensor([598,    22,   927,    77,  1191,   169,  1296,   316,  1430,   441,
                             1666,   852,  2367,  2487,  5530,  9213, 17114, 28437, 44424, 58356,
                             65832, 61560, 47884, 30933, 17030,  8114,  3425,  1231,   443,   120,
                             35,    12,     2], dtype=torch.int64)


class GNNTrainer(Base_Trainer):
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

        train_ind, val_ind = [], []
        for clss in range(4):
            random_ids = np.arange(clss*100, (clss+1)*100)
            np.random.shuffle(random_ids)
            train_ind += list(random_ids[:int(100*0.75)])
            val_ind += list(random_ids[int(100*0.75):])

        src_folder = os.path.join("data", "processed",
                                  "BACH_TRAIN")
        train_set, val_set = BACH(src_folder, ids=train_ind), BACH(src_folder, ids=val_ind)

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                                shuffle=False, num_workers=args["NUM_WORKERS"], persistent_workers=True)

        num_training_batches = len(train_loader)*args["EPOCHS"]

        print(f"Using {len(train_set)} training examples and {len(val_set)} validation example")

        model = SimpleGNN(img_size=args["IMG_SIZE"], num_batches=num_training_batches,
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
                             accumulate_grad_batches=64//args["BATCH_SIZE_TRAIN"],)

        ###########
        # EXTRAS  #
        ###########

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
