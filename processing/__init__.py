import numpy as np 
import networkx as nx 
import functools as ft 

from matplotlib import pyplot as plt 

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import stem_text, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_short

filters = [
    stem_text,                          # persons => person 
    strip_multiple_whitespaces,         # has    kelll => has kell  
    strip_tags,                         # <b>haskelll</b> => haslell
    strip_punctuation,                  # haskell! => haskell  
    ft.partial(strip_short, minsize=2)  # haskell is great  
]

def show_graph(G, tokens):
    labels = list(map(str, G.nodes()))
    labels = dict(zip(range(len(labels)), tokens))

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='r', node_size=300)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='b')
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.show()

def get_neighbors(index, sequence_length, winsize):
    assert winsize > 1 and winsize % 2 !=0  # winsize must be an odd number 
    offset = winsize // 2 
    indices = index + np.fromfunction(lambda i: i - offset, shape=[winsize])
    valid_mask = (indices >= 0) * (indices < sequence_length)
    return indices[valid_mask]

def to_graph(sequence, winsize=5):
    """ 7 // 2 => 3 
        LSTM RNN GRU 
        graph neural net 
        X, Y => X ===> Y 
        X : text => (map) => graph = G=(V, E)
        T = "......." => tokenized => [T0, T1, ..., TN]
        
        T0 => T1 => T2 => .... => TN 
        T0 => [T0, T1, T2]
        T1 => [T0, T1, T2, T3]
        T2 => [T0, T1, T2, T3, T4]

    """
    sequence_length = len(sequence)
    nodes = list(range(sequence_length))
    edges = []
    for idx in range(sequence_length):
        neighbors = get_neighbors(idx, sequence_length, winsize)
        for jdx in neighbors:
            edges.append( (idx, jdx) )
    
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes)
    return graph 

def process_one_text(text):  # closure 
    return preprocess_string(text, filters)