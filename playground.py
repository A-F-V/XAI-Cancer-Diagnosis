
from src.model.trainers.auto_encoder_trainer import CellAETrainer
import torch
from src.datasets.BACH_Cells import BACH_Cells
import os
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.autograd.set_detect_anomaly(True)
    #trainer = GNNTrainer()
    trainer = CellAETrainer()
    trainer.train()
    # BACH_Cells(os.path.join("data", "processed", "BACH_TRAIN")).compile_cells(recompute=True)
