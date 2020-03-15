import os
import sys

from cache import save_to_pickle
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import process_jsonl


def all_pipeline(jsonl_path, tokenize_fn):
    #  Read jsonl
    f = open(jsonl_path, "r")
    line_itr = f
    payload_saver = process_jsonl(line_itr, tokenize_fn)

    save_name = os.path.basename(jsonl_path)
    save_to_pickle(payload_saver, save_name)


if __name__ == "__main__":
    jsonl_path = sys.argv[1]
    voca_path = sys.argv[2]
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(jsonl_path, tokenize_fn)

