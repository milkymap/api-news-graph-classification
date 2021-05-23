import operator as op 
import functools as ft, itertools as it 

import dgl 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class MLP(nn.Module):
    """
        layers : array of layers List[int]
        activations : array of theta function : ['ReLu', 'ReLu', 'Softmax', 'Identity'] 
    """
    def __init__(self, layers, activations):
        super(MLP, self).__init__()  
        self.shapes = list(zip(layers[:-1], layers[1:]))
        self.layers = nn.ModuleList([ nn.Linear(m, n) for m,n in self.shapes ])
        self.thetas = []
        for f_name in activations:
            fn = op.attrgetter(f_name)(nn)() 
            self.thetas.append(fn)
    
    def forward(self, X):
        A = X
        for fn, layer in zip(self.thetas, self.layers):
            A = fn(layer(A))
        return A 

class Model(nn.Module):
    def __init__(self, vsize, hsize, asize, mlp_sizes, mlp_thetas):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vsize, embedding_dim=hsize)
        self.attention = dgl.nn.GATConv(hsize, asize, num_heads=3) 
        self.predictor = MLP(mlp_sizes, mlp_thetas)

    def forward(self, g, h0):
        """ 
            h0 : [f0, f1, ..., fn]
            h0 : [1, 9, 3, 0, 4, .... , 100]
        """ 
        h1 = self.embedding(h0)  # 1d => 2d (batch_size, hsize)
        h2 = th.mean(F.relu(self.attention(g, h1)), dim=1)  # (batch_size,asize)
        h3 = self.predictor(h2)
        g.ndata['h'] = h3 
        return dgl.mean_nodes(g, 'h')  # min, max, sum, mean 


if __name__ == '__main__':
    net = MLP([64, 32, 16, 4], ['ReLU', 'ReLU', 'Identity'])
    X = th.randn((10, 64))
    P = net(X)
    print(P)



        

         