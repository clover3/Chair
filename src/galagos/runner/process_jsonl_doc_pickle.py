import os
import sys

import datastore.tool
from cache import save_to_pickle
from cpath import data_path
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import process_jsonl


def all_pipeline(jsonl_path, tokenize_fn):
    #  Read jsonl
    f = open(jsonl_path, "r")
    line_itr = f
    buffered_saver = datastore.tool.PayloadSaver()
    payload_saver = process_jsonl(line_itr, tokenize_fn, buffered_saver)

    save_name = os.path.basename(jsonl_path)
    save_to_pickle(payload_saver, save_name)


if __name__ == "__main__":
    jsonl_path = sys.argv[1]
    if len(sys.argv) == 3:
        voca_path = sys.argv[2]
    else:
        voca_path = os.path.join(data_path, "bert_voca.txt")
        
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(jsonl_path, tokenize_fn)

