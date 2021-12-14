import click
from src.scripts.setup import setup
import os
from experiments.exp_code.monuseg_cell_binary import run as mcb_run


experiments = {"monuseg_cell_binary": mcb_run}


@click.command()
@click.argument('action')
@click.option('--model', '-m', default=None, help='Model to use')
@click.option('--experiment', '-e', default=None, help='Experiment to run')
def cli(action, model, experiment):
    if action == "setup":
        setup()
        return
    if action == "train":  # todo reselect API
        if model == None:
            print("Please specify a model to train")
            return
    if action == "run_experiment":
        if experiment == None or experiment not in experiments:
            print("Please specify a valid experiment to run")
            return
        experiments[experiment]()
        return
    print(f"{action} is not a valid argument")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cli()
