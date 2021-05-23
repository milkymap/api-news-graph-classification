from flask import Flask 
from os import getenv 

import pickle as pk 
import torch as th 

vocab_path = getenv('VOCAB_PATH')
model_path = getenv('MODEL_PATH')

if model_path is None: 
    print(' ... [model path is None] ... ')
    exit(1)

if vocab_path is None:
    print(' ... [vocab path is None] ... ')
    exit(1)


app = Flask(__name__)
app.config['brain'] = th.load(model_path)


with open(vocab_path, 'rb') as fp: 
    vocab = pk.load(fp)
    app.config['vocab'] = vocab 

print(app.config['brain'])
print(app.config['vocab'])

constants = {
    'server_up': {
        'status': 1, 
        'contents': 'The server is up ...!'
    },
    'invalid_request': {
        'status': 0, 
        'contents': 'Invalid request'
    }, 
    'prediction': {
        'status': 1,
        'contents': 'The guess was made'
    }, 
    'invalid_json_schema' : {
        'status': 0, 
        'content': 'The json schema should be {winsize: int, content: string}'
    }
}

from server import views 