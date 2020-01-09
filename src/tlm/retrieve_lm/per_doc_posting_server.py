import os
import pickle
import time

import cpath
from rpc.text_reader import *

PORT = 8126

def load_dict():
    st = time.time()
    token_path = os.path.join(cpath.data_path, "adhoc", "per_doc_posting.pickle")
    d = pickle.load(open(token_path, "rb"))

    for key,item in d.items():
        d[key] = dict(item)

    print(time.time() - st)
    return d


def start_server():
    server = TextReaderServer(load_dict)
    server.start(PORT)

def get_reader():
    return TextReaderClient(PORT)


if __name__ == "__main__":
    start_server()
