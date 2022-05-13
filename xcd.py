import click
import os

import json
from src.predict_cancer import predict_cancer


@click.command()
@click.argument('img_location')
@click.option('--concept_path', '-cp', default=os.path.join("data", "CONCEPTS_32"))
@click.option('--hovernet_path', '-hp', default=os.path.join("model", "HoVerNet.ckpt"))
@click.option('--cell_encoder_path', '-cep', default=os.path.join("model", "CellEncoder.ckpt"))
@click.option('--gnn_path', '-gp', default=os.path.join("model", "GCN.ckpt"))
@click.option('--explanation_file', '-eout', default=None)
def cli(img_location, concept_path, hovernet_path, cell_encoder_path, gnn_path, explanation_file):
    print("Predicting cancer for image: {}".format(img_location))
    prediction = predict_cancer(img_loc=img_location, hover_net_loc=hovernet_path, resnet_encoder=cell_encoder_path,
                                gnn_loc=gnn_path, explainability_location=explanation_file, concept_folder=concept_path)
    print("Prediction: {}".format(prediction))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cli()
