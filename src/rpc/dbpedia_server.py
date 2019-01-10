from rpc.text_reader import *
from data_generator.data_parser import trec
import path
import time
import os

def read_dbpedia():
    dbpedia_dir = os.path.join(path.data_path, "dbpedia")
    d_all = dict()
    for i in range(19):
        filename = "docs.{}.trectext".format(i)
        filepath = os.path.join(dbpedia_dir, filename)
        d = trec.load_trec(filepath, 1)
        d_all.update(d)

    return d_all


if __name__ == '__main__':
    begin = time.time()
    dicts = read_dbpedia()
    print(time.time() - begin, "elapsed")
    print("Total articles : " , len(dicts))

    server = TextReaderServer(read_dbpedia)
    server.start()
