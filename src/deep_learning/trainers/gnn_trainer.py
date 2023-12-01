import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.utils.util import wait_for_gpu
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
from ray import tune

from src.deep_learning.trainers.base_trainer import Base_Trainer
import os
from tqdm import tqdm
from torch_geometric.loader.dataloader import DataLoader
from src.transforms.image_processing.augmentation import StainJitter, RandomElasticDeformation, RandomFlip
import mlflow
from src.datasets.BACH import BACH, BACHSplitter
import pytorch_lightning as pl
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor, LambdaCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utilities.mlflow_utilities import log_plot
import numpy as np
from torchvision.transforms import RandomApply, RandomChoice
from src.deep_learning.architectures.cancer_prediction.cancer_gnn import CancerGNN
import json
import torch
from torch_geometric.transforms import Compose, KNNGraph, RandomJitter, Distance, RadiusGraph
from src.deep_learning.architectures.cancer_prediction.explainable_cancer_gnn import ExplainableCancerGNN
from src.deep_learning.architectures.components.graph_augmentation import NodePerturbation, NodeDropout
# p_mass=lambda x:far_mass((100/x)**0.5, 50, 0.001))
b_size = 32
pre_encoded = True

img_aug_train = Compose([
    RandomApply([RandomElasticDeformation(
        alpha=1.7, sigma=0.08, fields=['image'])], p=0.1),
    RandomApply(
        [
            StainJitter(theta=0.01, fields=["image"]),
            RandomFlip(fields=['image'])
        ],

        p=0.5),
    #  Normalize({"img": [0.6441, 0.4474, 0.6039]}, {"img": [0.1892, 0.1922, 0.1535]})
])
img_aug_val = Compose([])


def create_loaders(src_folder, train_ids, val_ids, **args):
    graph_aug_train = Compose([RandomJitter(30),
                               NodeDropout(0.9),
                               KNNGraph(k=args["K_NN"]),
                               NodePerturbation(0.15, 0.5)])
    graph_aug_val = Compose([KNNGraph(k=args["K_NN"])])

    train_set = BACH(src_folder, ids=train_ids,
                     graph_augmentation=graph_aug_train, img_augmentation=img_aug_train, pre_encoded=pre_encoded, preload=True)
    val_set = BACH(src_folder, ids=val_ids, graph_augmentation=graph_aug_val,
                   img_augmentation=img_aug_val, pre_encoded=pre_encoded, preload=True)

    # ensure that the intersection between train ids and val ids is 0
    assert len(set(train_ids).intersection(set(val_ids))) == 0

    print(f"Usig NUM WORKERS: {args['NUM_WORKERS']}")
    train_loader = DataLoader(train_set, batch_size=int(args["BATCH_SIZE_TRAIN"]),
                              shuffle=True, num_workers=args["NUM_WORKERS"], persistent_workers=True and args["NUM_WORKERS"] >= 1, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=int(args["BATCH_SIZE_VAL"]),
                            shuffle=False, num_workers=2, persistent_workers=True, drop_last=True)
    return train_loader, val_loader


def create_trainer(train_loader, val_loader, num_steps, accum_batch, grid_search=False, **args):
    ModelType = ExplainableCancerGNN if args["EXPLAINABLE"] else CancerGNN

    if args['START_CHECKPOINT'] is not None:
        model = ModelType.load_from_checkpoint(os.path.join(
            "experiments", "checkpoints", args['START_CHECKPOINT']), num_steps=num_steps, val_loader=val_loader, train_loader=train_loader, **args)
    else:
        model = ModelType(num_steps=num_steps,
                          val_loader=val_loader, train_loader=train_loader, pre_encoded=pre_encoded, **args)
    mlf_logger = MLFlowLogger(
        experiment_name=args["EXPERIMENT_NAME"], run_name=args["RUN_NAME"])

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

    trainer = pl.Trainer(log_every_n_steps=1,
                         max_epochs=args["EPOCHS"], logger=mlf_logger, callbacks=trainer_callbacks,
                         enable_checkpointing=not grid_search, default_root_dir=os.path.join("experiments", "checkpoints"),
                         profiler="simple",
                         accumulate_grad_batches=accum_batch, enable_progress_bar=not grid_search)
    return model, trainer


def cross_validate(src_folder, num_steps, accum_batch, k: int, **args):
    # split the dataset int k folds
    k_folds: list[list[int]] = BACHSplitter(src_folder).generate_k_folds(k)
    # for e ach fold, train and validate
    losses = []
    accuracies = []
    canc_accuracies = []
    for i in range(k):
        # Flatten the list of lists
        train_ids = [item for sublist in k_folds[:i] + k_folds[i+1:]
                     for item in sublist]

        val_ids = k_folds[i]

        train_loader, val_loader = create_loaders(
            src_folder, train_ids, val_ids, **args)
        model, trainer = create_trainer(
            train_loader, val_loader, num_steps, accum_batch, grid_search=False, **args)
        trainer.fit(model)

        # Calculate the metrics
        output = trainer.validate(model)
        losses.append(output[0]['ep/val_loss'])
        accuracies.append(output[0]['ep/val_acc'])
        canc_accuracies.append(output[0]['ep/val_canc_acc'])
    print(f"Losses: {losses}")
    print(f"Accuracies: {accuracies}")
    print(f"Canc Accuracies: {canc_accuracies}")


def grid_search(src_folder, num_steps, accum_batch, **args):
    # 1) Parse the grid search parameters
    def tuner_type_parser(tuner_info):  # expose outside
        if tuner_info["TYPE"] == "CHOICE":
            return tune.choice(tuner_info["VALUE"])
        if tuner_info["TYPE"] == "UNIFORM":
            return tune.uniform(*tuner_info["VALUE"])
        return None

    config = {tinfo["HP"]: tuner_type_parser(tinfo) for tinfo in args["GRID"]}
    # 2) Create the scheduler and reporter
    scheduler = ASHAScheduler(
        max_t=args["EPOCHS"],
        grace_period=args["EPOCHS"]//3,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "train_accuracy", "val_mean_accuracy", "train_canc_accuracy", "val_mean_canc_accuracy", "training_iteration"])

    def train_fn(config):
        targs = dict(args)
        targs.update(config)  # use args but with grid searched params

        # Create loader for just this trial
        train_ids, val_ids = BACHSplitter(
            src_folder).generate_train_val_split(0.8)
        train_loader, val_loader = create_loaders(
            src_folder, train_ids, val_ids, **targs)
        model, trainer = create_trainer(
            train_loader, val_loader, num_steps, accum_batch, grid_search=True, **targs)
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

    print(f"Best Config found was: " +
          str(analysis.get_best_config(metric="loss", mode="min")))


def make_checkpoint_path(file):
    print(file)
    return os.path.join("experiments", "checkpoints", file)


class GNNTrainer(Base_Trainer):
    def __init__(self, args=None):
        super(Base_Trainer, self).__init__()
        if args == None:
            args = json.load(
                open(os.path.join("experiments", "args", "gnn.json")))
        self.args = args

    def train(self, src_folder="C:\\Users\\aless\\Documents\\data"):

        print("Initializing Training")
        args = self.args
        print(f"The Args are: {args}")
        print("Getting the Data")

        run_name = str(args["RUN_NAME"])
        split_path = os.path.join(src_folder, f'graph_ind_{run_name}.txt')
        # Load saved inds
        if args["START_CHECKPOINT"]:
            train_ind, val_ind = BACHSplitter(
                src_folder).load_splits(split_path)
        else:
            train_ind, val_ind = BACHSplitter(
                src_folder).generate_train_val_split(0.8)

        print(f"The data source folder is {src_folder}")

        train_loader, val_loader = create_loaders(
            src_folder, train_ind, val_ind, **args)

        accum_batch = max(1, b_size//args["BATCH_SIZE_TRAIN"])
        num_steps = (len(train_loader)//accum_batch)*args["EPOCHS"]+1000

        print(
            f"Using {len(train_loader)} training examples and {len(val_loader)} validation example - With #{num_steps} steps")

        ###########
        # EXTRAS  #
        ###########

        ###########

        if args["LR_TEST"]:
            with mlflow.start_run(experiment_id=args["EXPERIMENT_ID"], run_name=args["RUN_NAME"]) as run:

                model, trainer = create_trainer(train_loader, val_loader, num_steps,
                                                accum_batch, grid_search=False, **args)
                lr_finder = trainer.tuner.lr_find(
                    model, num_training=args["EPOCHS"], max_lr=1000)
                fig = lr_finder.plot(suggest=True)
                log_plot(fig, "LR_Finder")
                print(lr_finder.suggestion())
        else:
            print("Training Started")
            if args["GRID_SEARCH"]:
                # grid search
                grid_search(src_folder, num_steps, accum_batch, **args)
            else:
                if (args["CROSS_VAL"]):
                    cross_validate(src_folder, num_steps,
                                   accum_batch, args["K_FOLDS"], **args)
                else:
                    model, trainer = create_trainer(
                        train_loader, val_loader, num_steps, accum_batch, **args)

                    trainer.fit(model)
                    # print("Training Over\nEvaluating")
                    # trainer.validate(model)
                    ckpt_file = str(args['EXPERIMENT_NAME']) + \
                        "_"+run_name+".ckpt"
                    ckpt_path = make_checkpoint_path(ckpt_file)
                    trainer.save_checkpoint(ckpt_path)

                    if (args["SAVE_IDS"]):
                        BACHSplitter(src_folder).save_split(
                            split_path, train_ind, val_ind)

                    return model, train_loader, val_loader

    def run(self, checkpoint):
        pass
