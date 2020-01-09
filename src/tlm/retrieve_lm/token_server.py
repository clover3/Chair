import os
import pickle

import cpath
from rpc.text_reader import *

PORT_TOKENREADER = 8124

def load_robust_token():
    token_path = os.path.join(cpath.data_path, "adhoc", "robust_tokens.pickle")
    return pickle.load(open(token_path, "rb"))

def start_token_server():
    server = TextReaderServer(load_robust_token)
    server.start(PORT_TOKENREADER)


def get_token_reader():
    return TextReaderClient(PORT_TOKENREADER)


if __name__ == "__main__":
    start_token_server()
