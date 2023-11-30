from src.deep_learning.trainers.gnn_trainer import GNNTrainer
import os

training_args = {
    "DEVICE": "cuda",
    "RUN_NAME": "FtT_30_11_2000",
    "EXPERIMENT_ID": 11,
    "EXPERIMENT_NAME": "FtT_Explainable",
    "EPOCHS": 2000,
    "BATCH_SIZE_TRAIN": 64,
    "BATCH_SIZE_VAL": 64,
    "MAX_LR": 1e-4,  # 0.0001
    "ONE_CYCLE": True,
    "START_LR": 1e-3,  # 0.000005
    "NUM_WORKERS": 4,
    "START_CHECKPOINT": None,
    "EARLY_STOP": False,
    "LR_TEST": False,
    "IMG_SIZE": 64,
    "INPUT_DROPOUT": 0.05,
    "L1_WEIGHT": 0.001,
    "GRID": [
        {
            "HP": "HEIGHT",
            "TYPE": "CHOICE",
            "VALUE": [4, 7, 9]
        },
        {
            "HP": "WIDTH",
            "TYPE": "CHOICE",
            "VALUE": [8, 16, 32, 64]
        },
        {
            "HP": "K_NN",
            "TYPE": "CHOICE",
            "VALUE": [4,  6]
        },
        {
            "HP": "INPUT_DROPOUT",
            "TYPE": "UNIFORM",
            "VALUE": [0, 0.3]
        }
    ],
    "GRID_SEARCH": False,
    "CROSS_VAL": False,
    "K_FOLDS": 4,
    "TRIALS": 15,
    "HEIGHT": 6,
    "WIDTH": 32,
    "CONCEPT_WIDTH": 64,
    "EXPLAINABLE": True,
    "K_NN": 5,
    "SAVE_IDS": True,
}

src_folder = os.path.join(
    "C://Users", "aless", "Documents", "FtT", "data", "BACH_TRAIN")
if __name__ == '__main__':
    trainer = GNNTrainer(training_args)
    trainer.train(src_folder=src_folder)
