import click 
import pickle as pk 

from torchtext.datasets import AG_NEWS 
from os import path 

@click.command()
@click.option('-t', '--target', help='target of data', required=True, type=str)
def grab_data(target):
    if not path.isfile(target):
        data = list(AG_NEWS(split='train'))  # download data 
        with open(target, 'wb') as fp: 
            pk.dump(data, fp)

if __name__ == '__main__':
    print(' ... [grabbing] ... ')
    grab_data()
    