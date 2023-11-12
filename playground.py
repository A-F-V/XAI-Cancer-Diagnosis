

import torch
import os
from src.datasets.BACH import BACH

from tqdm import tqdm
from src.predict_cancer import predict_cancer
from src.deep_learning.architectures.cancer_prediction.cell_encoder import CellEncoder
from src.predict_cancer import predict_cancer
import json


def create_encoded_graphs():

    src_folder = os.path.join(os.getcwd(), "data", "processed", "BACH_TRAIN")
    args = json.load(open(os.path.join(os.getcwd(), "experiments", "args", "default.json")))
    model = CellEncoder.load_from_checkpoint(os.path.join(
        os.getcwd(), "experiments", "checkpoints", "CellEncoder.ckpt"), train_loader=None, val_loader=None, data_set_path=src_folder, **args)
    with torch.no_grad():
        model.eval()
        model = model.cuda()

        bach_prob = BACH(src_folder, downsample=1)
        bach_prob.generate_encoded_graphs(model)


def create_test_set_predictions():
    directory = os.path.join("data", "raw", "unzipped", "BACH_TEST", "ICIAR2018_BACH_Challenge_TestDataset", "Photos")
    with open("predictions.csv", "w") as f:
        f.write("case,class,P(Normal),P(Benign),P(In Situ),P(Invasive)")
        for img_id in tqdm(range(100)):
            img_path = os.path.join(directory, f"test{img_id}.tif")
            prediction = 0
            probs = torch.as_tensor([0.3, 0.25, 0.25, 0.20])
            try:
                probs = predict_cancer(img_path).squeeze()
                probs_corrected = probs[[3, 0, 1, 2]]
                prediction = probs_corrected.argmax()+1
                assert prediction in [1, 2, 3, 4]

            except Exception as e:
                print(img_id)
                print(e)
                prediction = 0
            f.write(f"\n{img_id},{prediction},{','.join(map(lambda x:str(x.item()),list(probs)))}")


def test_explainability():
    directory = os.path.join("data", "raw", "unzipped", "BACH_TEST", "ICIAR2018_BACH_Challenge_TestDataset", "Photos")
    file_name = "test0.tif"
    explainability_path = os.getcwd()
    predict_cancer(os.path.join(directory, file_name), explainability_location=explainability_path)


def pre_encode_bach():
    node_embedder_model = CellEncoder.load_from_checkpoint(os.path.join("model", "CellEncoder.ckpt"))
    node_embedder_model.eval()
    node_embedder_model.requires_grad_(False)

    src_folder = "C:\\Users\\aless\\Documents\\data"
    BACH(src_folder, pre_encoded=False).generate_encoded_graphs(node_embedder_model)


def main():

    torch.multiprocessing.freeze_support()
    torch.autograd.set_detect_anomaly(True)

    # BACH_Cells(os.path.join("data", "processed", "BACH_TRAIN")).compile_cells(recompute=True, train_test_split=0.8)'

    # pre_encode_bach()
    #trainer = GNNTrainer()
    #trainer = CellAETrainer()
    #trainer = HoverNetTrainer()
    # trainer.train()

    # create_encoded_graphs()
    # create_test_set_predictions()
    # test_explainability()
    file = os.path.join("data", "raw", "unzipped", "BACH_TRAIN",
                        "ICIAR2018_BACH_Challenge", "Photos", "Benign", "b030.tif")
    predict_cancer(file, explainability_location="explain.png", concept_folder=os.path.join("data", "CONCEPTS_32"))


if __name__ == "__main__":
    main()
