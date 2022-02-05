from src.model.trainers.gnn_trainer import GNNTrainer
from src.model.trainers.auto_encoder_trainer import CellAETrainer
import torch
from src.datasets.BACH_Cells import BACH_Cells
import os
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    #trainer = GNNTrainer()
    trainer = CellAETrainer()
    trainer.train()
