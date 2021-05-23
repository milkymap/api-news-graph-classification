import click 
import pickle as pk 

import dgl 

from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

class Source(DGLDataset):
    def __init__(self, name, labels_endpoint, graphs_endpoint, vocab_endpoint):
        self.labels_endpoint = labels_endpoint
        self.graphs_endpoint = graphs_endpoint 
        self.vocab_endpoint = vocab_endpoint 
        super(Source, self).__init__(name)
    
    def process(self):
        with open(self.labels_endpoint, 'rb') as fp:
            self.labels = pk.load(fp)
        self.graphs, _ = dgl.load_graphs(self.graphs_endpoint)
        with open(self.vocab_endpoint, 'rb') as fp:
            self.vocab = pk.load(fp)
            
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = self.labels[idx]
        return g, y

@click.command()
@click.option('-n', '--name', help='name of the dataset', default='NEWS')
@click.option('-l', '--labels_endpoint', help='path to labels file', type=click.Path(True), required=True)
@click.option('-g', '--graphs_endpoint', help='path to graphs file', type=click.Path(True), required=True)
@click.option('-v', '--vocab_endpoint', help='path to vocab file', type=click.Path(True), required=True)
@click.option('-b', '--batch_size', help='size of the batch', type=int, default=8)
@click.pass_context
def create_source(ctx, name, labels_endpoint, graphs_endpoint, vocab_endpoint, batch_size):
    source = Source(name, labels_endpoint, graphs_endpoint, vocab_endpoint)
    loader = GraphDataLoader(dataset=source, batch_size=batch_size, shuffle=True)
    ctx.obj['source'] = source 
    ctx.obj['loader'] = loader  


if __name__ == '__main__':
    create_source()
  