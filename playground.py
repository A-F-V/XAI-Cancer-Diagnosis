
from src.model.trainers.auto_encoder_trainer import CellAETrainer
from src.model.trainers.gnn_trainer import GNNTrainer
import torch
from src.datasets.BACH_Cells import BACH_Cells
import os
from src.model.architectures.cancer_prediction.cell_unet_ae import UNET_AE
from src.datasets.BACH import BACH


def create_prob():
    from src.model.architectures.cancer_prediction.cell_unet_ae import UNET_AE
    import json
    src_folder = os.path.join(os.getcwd(), "data", "processed", "BACH_TRAIN")
    args = json.load(open(os.path.join(os.getcwd(), "experiments", "args", "default.json")))
    model = UNET_AE.load_from_checkpoint(os.path.join(
        os.getcwd(), "experiments", "checkpoints", "AE_UNET_PREDICTOR_NO_MSE.ckpt"), data_set_path=src_folder, **args)
    with torch.no_grad():
        model.eval()
        model = model.cuda()

        bach_prob = BACH(src_folder, downsample=1)
        bach_prob.generate_prob_graphs(model)


if __name__ == "__main__":

    torch.multiprocessing.freeze_support()
    torch.autograd.set_detect_anomaly(True)

    #BACH_Cells(os.path.join("data", "processed", "BACH_TRAIN")).compile_cells(recompute=True, train_test_split=0.8)

    trainer = GNNTrainer()
    #trainer = CellAETrainer()
    trainer.train()

    # create_prob()
