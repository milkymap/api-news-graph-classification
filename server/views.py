import json 
import dgl 

import torch as th

from flask import request 
from server import app, constants 
from processing import process_one_text, to_graph

@app.route('/main', methods=['GET'])
def main():
    return json.dumps(constants['server_up'])

@app.route('/guess', methods=['POST'])
def guess():
    try:
        app.config['brain'].eval()  # 
        req = request.json  
        
        if isinstance(req, dict):
            if 'winsize' in req and 'content' in req:
                # to sequence
                sequence = process_one_text(req['content'])
                print(sequence)
                # from [tok1, tok2, ..., tokN] => [i0, i1, ..., iN]
                indices = [ app.config['vocab'][tok] for tok in sequence ]
                # from [i0, i1, ..., iN] => graph 
                print(indices)
                graph = to_graph(sequence, req['winsize'])
                dg_graph = dgl.from_networkx(graph)
                dg_graph.ndata['h'] = th.as_tensor(indices).long()
                print(dg_graph)
                # make prediction 
                output = app.config['brain'](dg_graph, dg_graph.ndata['h'])
                flatten_output = th.squeeze(output) 
                print(output)
                argmax_index = th.argmax(flatten_output)
                print(argmax_index)
                response = {
                    'request_status': constants['prediction'],
                    'output': argmax_index.item()
                } 
                return json.dumps(response)
            else:
                return json.dumps(constants['invalid_json_schema'])
        else:
            pass 
    except Exception as e:
        print(e)
    return json.dumps(constants['invalid_request'])