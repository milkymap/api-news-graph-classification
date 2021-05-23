import click 
import torch as th 
import torch.nn as nn 
import torch.optim as optim 

from modelization.architecture import Model 
from modelization.producer import create_source 

from os import path

@click.group(invoke_without_command=True, chain=True)
@click.option('--debug/--no-debug', default=True)
@click.pass_context
def main(ctx, debug):
    if not ctx.invoked_subcommand:
        print(' ... [main] ... ') 
    ctx.obj['debug'] = debug 

@click.command()
@click.option('-e', '--epochs', help='number of epochs', default=100)
@click.option('-h', '--hsize', help='size of hidden', type=int)
@click.option('-a', '--asize', help='size of attention', type=int)
@click.option('-m', '--msize', help='size of the mlp', type=int, multiple=True)
@click.option('-t', '--theta', help='activations', type=str, multiple=True)
@click.option('-s', '--storage', help='path to model storage', type=click.Path(True))
@click.pass_context
def train(ctx, epochs, hsize, asize, msize, theta, storage):

    vsize = len(ctx.obj['source'].vocab)
    M = Model(vsize, hsize, asize, msize, theta)

    opt = optim.Adam(M.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() 
    for idx in range(epochs):
        cnt = 0
        for G, Y in ctx.obj['loader']:
            P = M(G, G.ndata['h'])
            E = criterion(P, Y)
            opt.zero_grad()
            E.backward()
            opt.step()
            cnt += 1
            print('Epoch : %03d | Index : %05d | Loss %07.4f' % (idx, cnt, E.item()))
    
    # end ...! 

    th.save(M, path.join(storage, 'brain.pt'))
    
# end ...! 

main.add_command(create_source)
main.add_command(train)

if __name__ == '__main__':
    print(' ... [learning] ... ')
    main(obj={})