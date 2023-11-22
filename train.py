from src.deep_learning.trainers.gnn_trainer import GNNTrainer
import os

training_args = {
    "DEVICE": "cuda",
    "RUN_NAME": "FtT_19_11_4",
    "EXPERIMENT_ID": 10,
    "EXPERIMENT_NAME": "FtT",
    "EPOCHS": 600,
    "BATCH_SIZE_TRAIN": 4,
    "BATCH_SIZE_VAL": 4,
    "MAX_LR": 1e-4,  # 0.0001
    "ONE_CYCLE": True,
    "START_LR": 1e-5,  # 0.000005
    "NUM_WORKERS": 1,
    "START_CHECKPOINT": None,
    "EARLY_STOP": False,
    "LR_TEST": False,
    "IMG_SIZE": 64,
    "INPUT_DROPOUT": 0.3,
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
    "K_FOLDS": 5,
    "TRIALS": 15,
    "HEIGHT": 7,
    "WIDTH": 100,
    "K_NN": 5,
    "SAVE_IDS": True,
}

src_folder = os.path.join(
    "C://Users", "aless", "Documents", "FtT", "data", "BACH_TRAIN")
if __name__ == '__main__':
    trainer = GNNTrainer(training_args)
    trainer.train(src_folder=src_folder)
