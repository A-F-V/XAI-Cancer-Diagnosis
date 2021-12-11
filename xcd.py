import click
from src.scripts.setup import setup
import os
from src.model.graph_construction.cell_segmentation import train as cs_train


@click.command()
@click.argument('action')
@click.option('--model', '-m', default=None, help='Model to use')
def cli(action, model):
    if action == "setup":
        setup()
        return
    if action == "train":
        if model == None:
            print("Please specify a model to train")
            return
        if model == "cell_seg":
            cs_train()
            return
    print(f"{action} is not a valid argument")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    cli()
