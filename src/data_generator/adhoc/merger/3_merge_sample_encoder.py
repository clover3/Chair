import os
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client
from data_generator.data_parser.trec import load_trec, load_robust
import random
import csv
from path import data_path, output_path
from data_generator.tokenizer_b import EncoderUnit
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from trainer.promise import PromiseKeeper, MyPromise, list_future
import pickle
import sys


def split_text(content):
    idx = 0
    window_size = 200 * 3
    while idx < len(content):
        span = content[idx:idx + window_size]
        yield span
        idx += window_size


def encode_payload(idx):
    doc_pairs = pickle.load(open("../output/merger_plainpair_{0:02d}.pickle".format(idx), "rb"))
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    max_sequence = 200
    encoder_unit = EncoderUnit(max_sequence, voca_path)

    result = []
    for q1, d1, q2, d2 in doc_pairs:
        enc_run_1 = [] 
        for text in split_text(d1):
            entry = encoder_unit.encode_pair(q1, text)
            enc_run_1.append((entry["input_ids"], entry["input_mask"], entry["segment_ids"]))

        enc_run_2 = []
        for text in split_text(d2):
            entry = encoder_unit.encode_pair(q2, text)
            enc_run_2.append((entry["input_ids"], entry["input_mask"], entry["segment_ids"]))
        result.append((enc_run_1, enc_run_2))

    filename = os.path.join(output_path, "merger_train_{}.pickle".format(idx))
    pickle.dump(result, open(filename, "wb"))


if __name__ == '__main__':
    encode_payload(int(sys.argv[1]))

