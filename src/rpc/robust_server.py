import time

from data_generator.data_parser.trec import load_robust_ingham
from rpc.text_reader import *

if __name__ == '__main__':
    begin = time.time()
    server = TextReaderServer(load_robust_ingham)
    server.start()
