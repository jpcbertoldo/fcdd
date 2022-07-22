"""
this is a hacking of $(which wandb)
"""
import sys
from wandb.cli.cli import cli
if __name__ == '__main__':
    sys.exit(cli())
