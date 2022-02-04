from src.model.trainers.gnn_trainer import GNNTrainer
from src.model.trainers.auto_encoder_trainer import CellAETrainer
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    #trainer = GNNTrainer()
    trainer = CellAETrainer()
    trainer.train()
    
