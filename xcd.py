import click
from src.scripts.setup import setup
import os


@click.command()
@click.argument('action')
def cli(action):
    if action == "setup":
        setup()
        return
    print(f"{action} is not a valid argument")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    cli()
