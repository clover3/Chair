from rpc.text_reader import *
from data_generator.data_parser.trec import load_robust_ingham
import path
import time
import os



if __name__ == '__main__':
    begin = time.time()
    server = TextReaderServer(load_robust_ingham)
    server.start()
