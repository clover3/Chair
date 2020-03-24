from __future__ import absolute_import

import sys

import datastore.interface
import datastore.tool
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import parse_doc_and_save
from galagos.interface import get_doc


def all_pipeline(tokenize_fn, index_path, doc_id):
    #  Read jsonl
    buffered_saver = datastore.tool.BufferedSaver()
    html = get_doc(index_path, doc_id)
    parse_doc_and_save(buffered_saver, doc_id, html, tokenize_fn)
    buffered_saver.flush()
    print("Done")


if __name__ == "__main__":
    print("get_doc_and_process")
    voca_path = sys.argv[1]
    index_path = sys.argv[2]
    doc_id = sys.argv[3]
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(tokenize_fn, index_path, doc_id)

