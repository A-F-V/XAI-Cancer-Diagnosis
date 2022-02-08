import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.utils.util import wait_for_gpu
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
from ray import tune
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
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate

from src.transforms.graph_augmentation.edge_dropout import EdgeDropout, far_mass

# p_mass=lambda x:far_mass((100/x)**0.5, 50, 0.001))


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

        graph_aug_train = Compose([RandomTranslate(30), KNNGraph(k=6), EdgeDropout(p=0.04)]
                                  )  # !TODO RECOMPUTE EDGE WEIGHTS
        graph_aug_pred = Compose([KNNGraph(k=6)])

        train_ind, val_ind = [], []
        for clss in range(4):
            random_ids = np.arange(clss*100, (clss+1)*100)
            np.random.shuffle(random_ids)
            train_ind += list(random_ids[:int(100*0.75)])
            val_ind += list(random_ids[int(100*0.75):])

        src_folder = os.path.join(os.getcwd(), "data", "processed",
                                  "BACH_TRAIN")
        print(f"The data source folder is {src_folder}")
        train_set, val_set = BACH(src_folder, ids=train_ind,
                                  graph_augmentation=graph_aug_train), BACH(src_folder, ids=val_ind, graph_augmentation=graph_aug_pred)

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True if args["NUM_WORKERS"] > 0 else False)
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                                shuffle=False, num_workers=args["NUM_WORKERS"], persistent_workers=True if args["NUM_WORKERS"] > 0 else False)

        accum_batch = max(1, 64//args["BATCH_SIZE_TRAIN"])
        num_steps = len(train_loader)*args["EPOCHS"]//accum_batch

        print(f"Using {len(train_set)} training examples and {len(val_set)} validation example - With #{num_steps} steps")

        ###########
        # EXTRAS  #
        ###########

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:

                model, trainer = create_trainer(train_loader, val_loader, num_steps,
                                                accum_batch, grid_search=False, **args)
                lr_finder = trainer.tuner.lr_find(model, num_training=1000, max_lr=0.01)
                fig = lr_finder.plot(suggest=True)
                log_plot(fig, "LR_Finder")
                print(lr_finder.suggestion())
        else:
            print("Training Started")
            if args["GRID_SEARCH"]:
                # grid search
                grid_search(train_loader, val_loader, num_steps, accum_batch, **args)
            else:
                model, trainer = create_trainer(train_loader, val_loader, num_steps, accum_batch, **args)

                trainer.fit(model)
                # print("Training Over\nEvaluating")
                # trainer.validate(model)
                ckpt_file = str(args['EXPERIMENT_NAME'])+"_"+str(args['RUN_NAME'])+".ckpt"
                ckpt_path = make_checkpoint_path(ckpt_file)
                trainer.save_checkpoint(ckpt_path)

    def run(self, checkpoint):
        pass


def grid_search(train_loader, val_loader, num_steps, accum_batch, **args):
    def tuner_type_parser(tuner_info):  # expose outside
        if tuner_info["TYPE"] == "CHOICE":
            return tune.choice(tuner_info["VALUE"])
        if tuner_info["TYPE"] == "UNIFORM":
            return tune.uniform(*tuner_info["VALUE"])
        return None

    config = {tinfo["HP"]: tuner_type_parser(tinfo) for tinfo in args["GRID"]}
    scheduler = ASHAScheduler(
        max_t=args["EPOCHS"],
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "mean_accuracy", "mean_canc_accuracy", "training_iteration"])

    def train_fn(config):
        wait_for_gpu(target_util=0.1)
        targs = dict(args)
        targs.update(config)  # use args but with grid searched params
        model, trainer = create_trainer(train_loader, val_loader, num_steps, accum_batch, grid_search=True, ** targs)
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


def create_trainer(train_loader, val_loader, num_steps, accum_batch, grid_search=False, **args):
    model = SimpleGNN(img_size=args["IMG_SIZE"], num_steps=num_steps,
                      val_loader=val_loader, train_loader=train_loader, **args)
    mlf_logger = MLFlowLogger(experiment_name=args["EXPERIMENT_NAME"], run_name=args["RUN_NAME"])

    trainer_callbacks = [
    ]

    if args["LR_TEST"]:
        trainer_callbacks.append(LearningRateMonitor(logging_interval='step'))
    if args["EARLY_STOP"]:
        trainer_callbacks.append(EarlyStopping(monitor="ep/val_loss"))

    if grid_search:
        trc = TuneReportCallback(
            {
                "loss": "ep/val_loss",
                "mean_accuracy": "ep/val_acc",
                "mean_canc_accuracy": "ep/val_canc_acc"
            },
            on="validation_end")
        trainer_callbacks.append(trc)

    trainer = pl.Trainer(log_every_n_steps=1, gpus=1,
                         max_epochs=args["EPOCHS"], logger=mlf_logger if not grid_search else False, callbacks=trainer_callbacks,
                         enable_checkpointing=not grid_search, default_root_dir=os.path.join("experiments", "checkpoints"),
                         profiler="simple",
                         accumulate_grad_batches=accum_batch, progress_bar_refresh_rate=0 if grid_search else 1)
    return model, trainer


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)
