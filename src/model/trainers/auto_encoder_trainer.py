from ast import arg
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.utils.util import wait_for_gpu
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
from ray import tune
from src.model.architectures.cancer_prediction.pred_gnn import PredGNN
from src.model.trainers.base_trainer import Base_Trainer
import os
from tqdm import tqdm
from torch_geometric.loader.dataloader import DataLoader
from src.transforms.image_processing.augmentation import *
import mlflow
from src.datasets.BACH import BACH
import pytorch_lightning as pl
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor, LambdaCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utilities.mlflow_utilities import log_plot
import numpy as np
from src.model.architectures.cancer_prediction.cancer_net import CancerNet
from src.model.architectures.cancer_prediction.cancer_predictor import CancerPredictorGNN
import json
import torch
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate, Distance
from src.datasets.BACH_Cells import BACH_Cells
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.transforms.graph_augmentation.edge_dropout import EdgeDropout, far_mass
from src.transforms.graph_augmentation.largest_component import LargestComponent
from src.model.architectures.cancer_prediction.cell_encoder import CellEncoder
from torchvision.transforms import RandomApply, RandomChoice

# p_mass=lambda x:far_mass((100/x)**0.5, 50, 0.001))


# Normalize({"image": [0.6441, 0.4474, 0.6039]},{"image": [0.1892, 0.1922, 0.1535]})   # for image not optical
batch_size = 256

transforms_training = Compose([

    StainJitter(theta=0.025, fields=["img"]),
    RandomFlip(fields=['img']),
    RandomApply(
        [

            # RandomChoice([
            #    AddGaussianNoise(0.01, fields=["img"]),
            #    GaussianBlur(fields=["img"])]),
            RandomElasticDeformation(alpha=1.7, sigma=0.08, fields=['img'])

            # ColourJitter(bcsh=(0.2, 0.1, 0.1, 0.1), fields=["image"]),
        ],

        p=0.8),
    #  Normalize({"img": [0.6441, 0.4474, 0.6039]}, {"img": [0.1892, 0.1922, 0.1535]})
])

transforms_val = Compose([
    # Normalize({"img": [0.6441, 0.4474, 0.6039]}, {"img": [0.1892, 0.1922, 0.1535]})
])


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

        src_folder = "C:\\Users\\aless\\Documents\\data"
        args["SRC_FOLDER"] = src_folder

        #############

       # print(f"Using {len(train_set)} training examples and {len(val_set)} validation example - With #{num_steps} steps")

        ###########
        # EXTRAS  #
        ###########

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:

                model, trainer = create_trainer(grid_search=False, **args)
                lr_finder = trainer.tuner.lr_find(model, num_training=1500, max_lr=1000)
                fig = lr_finder.plot(suggest=True)
                log_plot(fig, "LR_Finder")
                print(lr_finder.suggestion())
        else:
            print("Training Started")
            if args["GRID_SEARCH"]:
                # grid search
                grid_search(**args)
            else:
                model, trainer = create_trainer(**args)

                trainer.fit(model)
                # print("Training Over\nEvaluating")
                # trainer.validate(model)
                ckpt_file = str(args['EXPERIMENT_NAME'])+"_"+str(args['RUN_NAME'])+".ckpt"
                ckpt_path = make_checkpoint_path(ckpt_file)
                trainer.save_checkpoint(ckpt_path)

    def run(self, checkpoint):
        pass


def grid_search(**args):
    def tuner_type_parser(tuner_info):  # expose outside
        if tuner_info["TYPE"] == "CHOICE":
            return tune.choice(tuner_info["VALUE"])
        if tuner_info["TYPE"] == "UNIFORM":
            return tune.uniform(*tuner_info["VALUE"])
        return None

    config = {tinfo["HP"]: tuner_type_parser(tinfo) for tinfo in args["GRID"]}
    scheduler = ASHAScheduler(
        max_t=args["EPOCHS"],
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "mean_accuracy", "mean_canc_accuracy", "training_iteration"])

    def train_fn(config):
        # wait_for_gpu(target_util=0.1)
        targs = dict(args)
        targs.update(config)  # use args but with grid searched params
        model, trainer = create_trainer(grid_search=True, ** targs)
        trainer.fit(model)

    resources_per_trial = {"cpu": 1, "gpu": 1}
    analysis = tune.run(train_fn,
                        metric="loss",
                        mode="min",
                        config=config,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name="tune_gnn_asha",
                        resources_per_trial=resources_per_trial,
                        num_samples=args["TRIALS"])  # number of trials

    print(f"Best Config found was: " + analysis.get_best_config(metric="loss", mode="min"))


def create_trainer(grid_search=False, **args):

    src_folder = args["SRC_FOLDER"]
    # train_set, val_set = train_val_split(BACH_Cells, src_folder, 0.8, tr_trans=tr_trans, val_trans=val_trans)
    train_set, val_set = BACH_Cells(src_folder, transform=transforms_training, val=False), BACH_Cells(
        src_folder, transform=transforms_val, val=True)

    train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                              shuffle=True, num_workers=args["NUM_WORKERS"],
                              persistent_workers=True
                              )
    val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                            shuffle=False, num_workers=args["NUM_WORKERS"],
                            persistent_workers=True)

    accum_batch = max(1, batch_size//args["BATCH_SIZE_TRAIN"])
    num_steps = (len(train_loader)//accum_batch+1)*args["EPOCHS"]

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer_callbacks = [
        lr_monitor
    ]
    if args["START_CHECKPOINT"] is not None:
        model = CellEncoder.load_from_checkpoint(os.path.join('experiments', 'checkpoints', args["START_CHECKPOINT"]), train_loader=train_loader,
                                                 val_loader=val_loader, img_size=64, num_steps=num_steps, **args)
    else:
        model = CellEncoder(train_loader=train_loader, val_loader=val_loader, img_size=64, num_steps=num_steps, **args)
    mlf_logger = MLFlowLogger(experiment_name=args["EXPERIMENT_NAME"], run_name=args["RUN_NAME"])

    # model.encodercnn.requires_grad_(False)

    trainer = pl.Trainer(log_every_n_steps=1, gpus=1, callbacks=trainer_callbacks,
                         max_epochs=args["EPOCHS"], logger=mlf_logger,
                         enable_checkpointing=not grid_search, default_root_dir=os.path.join("experiments", "checkpoints"),
                         profiler="simple",
                         accumulate_grad_batches=accum_batch, enable_progress_bar=not grid_search)
    return model, trainer


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)
