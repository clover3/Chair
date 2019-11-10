from rpc.text_reader import *
import pickle
from misc_lib import TimeEstimator
from data_generator.data_parser.trec import load_robust_ingham, load_robust
import path
import time
import os
from data_generator import tokenizer_b as tokenization


PORT_TOKENREADER = 8124


def gen_robust_token():
    collection = load_robust_ingham()

    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    ticker = TimeEstimator(len(collection))
    nc = {}
    for doc_id, content in collection.items():
        nc[doc_id] = tokenizer.basic_tokenizer.tokenize(content)
        ticker.tick()

    token_path = os.path.join(path.data_path, "adhoc", "robust_tokens.pickle")
    pickle.dump(nc, open(token_path, "wb"))





def load_robust_token():
    token_path = os.path.join(path.data_path, "adhoc", "robust_tokens.pickle")
    return pickle.load(open(token_path, "rb"))


def start_token_server():
    server = TextReaderServer(load_robust_token)
    server.start(PORT_TOKENREADER)


def get_token_reader():
    return TextReaderClient(PORT_TOKENREADER)


if __name__ == "__main__":
    start_token_server()
