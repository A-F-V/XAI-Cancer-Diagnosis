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
from pytorch_lightning.callbacks import LearningRateMonitor, LambdaCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utilities.mlflow_utilities import log_plot
import numpy as np


from src.model.architectures.cancer_prediction.cancer_gnn import CancerGNN
import json
import torch
from torch_geometric.transforms import Compose, KNNGraph, RandomTranslate, Distance

from src.transforms.graph_augmentation.edge_dropout import EdgeDropout, far_mass
from src.transforms.graph_augmentation.largest_component import LargestComponent

# p_mass=lambda x:far_mass((100/x)**0.5, 50, 0.001))
b_size = 256


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

        graph_aug_train = Compose([RandomTranslate(40), KNNGraph(k=args["K_NN"]),   Distance(norm=False, cat=False)]  # EdgeDropout(p=0.04),
                                  )
        graph_aug_pred = Compose([KNNGraph(k=args["K_NN"]),  Distance(norm=False, cat=False)])

        train_ind, val_ind = [], []
        src_folder = "C:\\Users\\aless\\Documents\\data"
        graph_split = os.path.join(src_folder, "graph_ind.txt")
        with open(graph_split, "r") as f:
            l1 = f.readline().strip()
            l2 = f.readline().strip()
            train_ind = list(map(int, l1[1:-1].split(",")))
            val_ind = list(map(int, l2[1:-1].split(",")))
            # for clss in range(4):
            #    random_ids = np.arange(clss*100, (clss+1)*100)
            #    np.random.shuffle(random_ids)
            #    train_ind += list(random_ids[:int(100*0.75)])
            #    val_ind += list(random_ids[int(100*0.75):])

        #train_ind = list(range(400))
        print(f"The data source folder is {src_folder}")
        train_set, val_set = BACH(src_folder, ids=train_ind,
                                  graph_augmentation=graph_aug_train), BACH(src_folder, ids=val_ind, graph_augmentation=graph_aug_pred)

        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE_TRAIN"],
                                  shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True if args["NUM_WORKERS"] > 0 else False)
        val_loader = DataLoader(val_set, batch_size=args["BATCH_SIZE_VAL"],
                                shuffle=False, num_workers=args["NUM_WORKERS"], persistent_workers=True if args["NUM_WORKERS"] > 0 else False)

        accum_batch = max(1, b_size//args["BATCH_SIZE_TRAIN"])
        num_steps = (len(train_loader)//accum_batch+1)*args["EPOCHS"]

        print(f"Using {len(train_set)} training examples and {len(val_set)} validation example - With #{num_steps} steps")

        ###########
        # EXTRAS  #
        ###########

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:

                model, trainer = create_trainer(train_loader, val_loader, num_steps,
                                                accum_batch, grid_search=False, **args)
                lr_finder = trainer.tuner.lr_find(model, num_training=300, max_lr=1000)
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
        grace_period=40,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "train_accuracy", "val_mean_accuracy", "train_canc_accuracy", "val_mean_canc_accuracy", "training_iteration"])

    def train_fn(config):
        # wait_for_gpu(target_util=0.1)
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

    print(f"Best Config found was: " + str(analysis.get_best_config(metric="loss", mode="min")))


def create_trainer(train_loader, val_loader, num_steps, accum_batch, grid_search=False, **args):
    model = CancerGNN(num_steps=num_steps,
                      val_loader=val_loader, train_loader=train_loader, **args)
    mlf_logger = MLFlowLogger(experiment_name=args["EXPERIMENT_NAME"], run_name=args["RUN_NAME"])

    ############################
    # Layering
    ###################
    # model.layers = 1

    def layer_after_x(time):
        def _layer(trainer, pl_module):
            if trainer.current_epoch >= time:
                model.layers = args["LAYERS"]
        return _layer

    ############################

    trainer_callbacks = [
        # LambdaCallback(on_train_epoch_start=unfreeze_after_x("steepness", 50))
        # LambdaCallback(on_train_epoch_start=layer_after_x(10))
    ]

    trainer_callbacks.append(LearningRateMonitor(logging_interval='step'))
    if args["EARLY_STOP"]:
        trainer_callbacks.append(EarlyStopping(monitor="ep/val_loss"))

    if grid_search:
        trc = TuneReportCallback(
            {
                "loss": "ep/val_loss",
                "val_mean_accuracy": "ep/val_acc",
                "val_mean_canc_accuracy": "ep/val_canc_acc",

            },
            on="validation_end")
        trct = TuneReportCallback(
            {
                "train_canc_accuracy": "ep/train_canc_acc",
                "train_accuracy": "ep/train_acc"
            },
            on="train_end")
        trainer_callbacks.append(trc)
        trainer_callbacks.append(trct)

    trainer = pl.Trainer(log_every_n_steps=1, gpus=1,
                         max_epochs=args["EPOCHS"], logger=mlf_logger, callbacks=trainer_callbacks,
                         enable_checkpointing=not grid_search, default_root_dir=os.path.join("experiments", "checkpoints"),
                         profiler="simple",
                         accumulate_grad_batches=accum_batch, enable_progress_bar=not grid_search)
    return model, trainer


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)
