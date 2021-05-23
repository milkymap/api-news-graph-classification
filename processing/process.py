import click 
import pickle as pk 
import dgl 

import torch as th 

from collections import Counter  
from torchtext.vocab import Vocab  
from processing import process_one_text, to_graph, show_graph
from os import path 

@click.command()
@click.option('-s', '--source', help='path to source data', type=click.Path(True), required=True)
@click.option('-o', '--offset', help='offset on dataset', type=click.IntRange(0, 120000), default=0)
@click.option('-n', '--nb_samples', help='number of samples', type=int, default=10000)
@click.option('-w', '--winsize', help='size of the (neighbors) window', default=5, type=int)
@click.option('-d', '--dump', help='feature storage', type=click.Path(True))
def process_data(source, offset, nb_samples, winsize, dump):
    with open(source, 'rb') as fp: 
        data = pk.load(fp)
    
    counter = Counter()
    tokens_acc = []
    labels_acc = []
    graphs_acc = []
    for idx, (label, contents) in enumerate(data[offset:offset+nb_samples]): 
        print('processing of the text number : %05d ... ' % idx)
        tokens = process_one_text(contents)
        tokens_acc.append(tokens)
        counter.update(tokens)  # tok : frequence 
        labels_acc.append(label - 1)  # 1 2 3 4 => 0 1 2 3 

    vocab = Vocab(counter)
    print('size of vocab %04d' % len(vocab))

    for idx, tokens in enumerate(tokens_acc):
        print('generation graph n° %05d' % idx)
        tok_idx = [ vocab[tok] for tok in tokens ]
        graph = to_graph(tokens, winsize)
        dg_graph = dgl.from_networkx(graph)
        dg_graph.ndata['h'] = th.as_tensor(tok_idx).long()
        graphs_acc.append(dg_graph)
    
    with open(path.join(dump, 'news.pkl'), 'wb') as fp:
        pk.dump(labels_acc, fp)
    
    with open(path.join(dump, 'vocab.pkl'), 'wb') as fp:
        pk.dump(vocab, fp)
    
    dgl.save_graphs(path.join(dump, 'news.dgl'), graphs_acc)

if __name__ == '__main__':
    print(' ... [processing] ... ')
    process_data()