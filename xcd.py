import click
from src.scripts.setup import setup
import os
from src.model.trainers.hover_net_trainer import HoverNetTrainer
import json
from src.model.trainers.gnn_trainer import GNNTrainer

models = {"hover_net": HoverNetTrainer, "gnn": GNNTrainer}


@click.command()
@click.argument('action')
@click.option('--model', '-m', default=None, help='Model to use')
@click.option('--checkpoint', '-c', default=None, help='Checkpoint to use')
@click.option('--args', default=os.path.join("experiments", "args", "default.json"), help="File containing args")
def cli(action, model, checkpoint, args):
    args = json.load(open(args))
    if action == "setup":
        setup()
        return
    if action == "train":
        if model == None or model not in models.keys():
            print(f"Please specify a model from {models.keys()} to train")
            return
        models[model](args).train()
        return
    if action == "run":
        if model == None or model not in models.keys():
            print(f"Please specify a model from {models.keys()} to run")
            return
        models[model](args).run(checkpoint)
        return
    # if action == "run_experiment":
    #    if experiment == None or experiment not in experiments:
    #        print("Please specify a valid experiment to run")
    #        return
    #    experiments[experiment]()
    #    return
    print(f"{action} is not a valid argument")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cli()
