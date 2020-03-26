from __future__ import absolute_import

import sys

import datastore.interface
import datastore.tool
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import process_jsonl
from misc_lib import file_iterator_interval


def all_pipeline(jsonl_path, tokenize_fn, task_idx):
    #  Read jsonl
    f = open(jsonl_path, "r")
    block = 10 * 1000
    st =  task_idx * block
    ed = (task_idx+1) * block
    line_itr = file_iterator_interval(f, st, ed)
    buffered_saver = datastore.tool.PayloadSaver()
    process_jsonl(line_itr, tokenize_fn, buffered_saver)


if __name__ == "__main__":
    jsonl_path = sys.argv[1]
    voca_path = sys.argv[2]
    task_idx = int(sys.argv[3])
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(jsonl_path, tokenize_fn, task_idx)

