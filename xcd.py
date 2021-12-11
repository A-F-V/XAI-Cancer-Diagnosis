import click
from src.scripts.setup import setup


@click.command()
@click.argument('action')
def cli(action):
    if action == "setup":
        setup()
        return
    print(f"{action} is not a valid argument")


if __name__ == '__main__':
    cli()
