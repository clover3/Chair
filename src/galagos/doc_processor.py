from __future__ import absolute_import

import sys

from boilerpipe.extract import Extractor
from nltk import tokenize

import datastore.interface
from data_generator.tokenizer_wo_tf import FullTokenizer
from datastore.table_names import *
from galagos.basic import parse_doc_jsonl_line
from misc_lib import TimeEstimator


def all_pipeline(jsonl_path, tokenize_fn):
    #  Read jsonl
    f = open(jsonl_path, "r")
    ticker = TimeEstimator(100*1000)
    for line in f:
        doc_id, html = parse_doc_jsonl_line(line)
        try:
            # remove boilderplate
            datastore.interface.save(RawCluewebDoc, doc_id, html)
            extractor = Extractor(extractor='ArticleExtractor', html=html)
            core_text = extractor.getText()

            # write boilerplate removed data as jsonl
            data = {
                    "doc_id": doc_id,
                    "content": core_text
                    }
            datastore.interface.save(CleanedCluewebDoc, doc_id, html)

            # tokenize the docs
            chunks = []
            for sent in tokenize.sent_tokenize(core_text):
                tokens = tokenize_fn(sent)
                chunks.append(tokens)
            datastore.interface.save(BertTokenizedCluewebDoc, doc_id, chunks)

            tokens = tokenize.word_tokenize(core_text)
            datastore.interface.save(TokenizedCluewebDoc, doc_id, tokens)
            ticker.tick()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    jsonl_path = sys.argv[1]
    voca_path = sys.argv[2]
    tokenize_fn = FullTokenizer(voca_path, True).tokenize
    all_pipeline(jsonl_path, tokenize_fn)

