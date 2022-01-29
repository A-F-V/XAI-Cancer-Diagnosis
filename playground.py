from src.model.trainers.gnn_trainer import GNNTrainer
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    trainer = GNNTrainer()
    trainer.train()
