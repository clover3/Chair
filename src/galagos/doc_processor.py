from __future__ import absolute_import

import sys
from collections import Counter

from boilerpipe.extract import Extractor
from nltk import tokenize

import datastore.interface
import datastore.tool
from data_generator.tokenizer_wo_tf import FullTokenizer
from datastore.table_names import *
from galagos.basic import parse_doc_jsonl_line
from misc_lib import TimeEstimator


def file_iterator_interval(f, st, ed):
    for idx, line in enumerate(f):
        if idx < st:
            pass
        elif idx < ed:
            yield line
        else:
            break


def all_pipeline(jsonl_path, tokenize_fn, task_idx):
    #  Read jsonl
    f = open(jsonl_path, "r")
    block = 10 * 1000
    st =  task_idx * block
    ed = (task_idx+1) * block
    line_itr = file_iterator_interval(f, st, ed)
    process_jsonl(line_itr, tokenize_fn)


def process_jsonl(line_itr, tokenize_fn):
    ticker = TimeEstimator(5*1000)
    buffered_saver = datastore.tool.PayloadSaver()
    for line in line_itr:
        doc_id, html = parse_doc_jsonl_line(line)
        ticker.tick()
        if datastore.interface.has_key(RawCluewebDoc, doc_id):
            continue
        try:
            # remove boilderplate
            buffered_saver.save(RawCluewebDoc, doc_id, html)
            extractor = Extractor(extractor='ArticleExtractor', html=html)
            core_text = extractor.getText()

            # write boilerplate removed data as jsonl
            data = {
                "doc_id": doc_id,
                "content": core_text
            }
            buffered_saver.save(CleanedCluewebDoc, doc_id, html)

            # tokenize the docs
            chunks = []
            for sent in tokenize.sent_tokenize(core_text):
                tokens = tokenize_fn(sent)
                chunks.append(tokens)
            buffered_saver.save(BertTokenizedCluewebDoc, doc_id, chunks)

            tokens = tokenize.word_tokenize(core_text)
            buffered_saver.save(TokenizedCluewebDoc, doc_id, tokens)

            tf = Counter(tokens)
            buffered_saver.save(CluewebDocTF, doc_id, tf)
        except Exception as e:
            print(e)

    return buffered_saver

if __name__ == "__main__":
    jsonl_path = sys.argv[1]
    voca_path = sys.argv[2]
    task_idx = int(sys.argv[3])
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(jsonl_path, tokenize_fn, task_idx)

