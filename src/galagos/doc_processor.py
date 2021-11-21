from __future__ import absolute_import

import time
from collections import Counter
from typing import List, Tuple

from boilerpipe.extract import Extractor
from nltk import tokenize

import datastore.interface
import datastore.tool
from data_generator.tokenize_helper import TokenizedText
from datastore.table_names import *
from galagos.parse import parse_doc_jsonl_line
from galagos.tokenize_doc_and_save import bert_tokenize
from misc_lib import TimeEstimator


def ask_key(doc_id):
    st = time.time()
    r = datastore.interface.has_key(RawCluewebDoc, doc_id)
    ed = time.time()

    if ed - st > 5:
        print("Asking takes too long")
    return r


def process_jsonl(line_itr, tokenize_fn, buffered_saver, num_insts=0):
    if num_insts:
        ticker = TimeEstimator(num_insts)

    for line in line_itr:
        doc_id, html = parse_doc_jsonl_line(line)
        if num_insts:
            ticker.tick()
        try:
            # remove boilderplate
            parse_doc_and_save(buffered_saver, doc_id, html, tokenize_fn)
        except Exception as e:
            print("Exception at parse_doc_and_save")
            print(e)

    return buffered_saver


def parse_doc_and_save(buffered_saver, doc_id, html, tokenize_fn):
    buffered_saver.save(RawCluewebDoc, doc_id, html)
    extractor = Extractor(extractor='ArticleExtractor', html=html)
    core_text = extractor.getText()
    core_text = str(core_text)
    # write boilerplate removed data as jsonl
    buffered_saver.save(CleanedCluewebDoc, doc_id, html)
    # tokenize the docs
    chunks = bert_tokenize(core_text, tokenize_fn)
    buffered_saver.save(BertTokenizedCluewebDoc, doc_id, chunks)
    tokens = tokenize.word_tokenize(core_text)
    buffered_saver.save(TokenizedCluewebDoc, doc_id, tokens)
    tf = Counter(tokens)
    buffered_saver.save(CluewebDocTF, doc_id, tf)


class BoilerpipeException(Exception):
    pass


def jsonl_to_tokenized_text(line_itr, tokenizer, num_insts=0) -> List[Tuple[str, TokenizedText]]:
    if num_insts:
        ticker = TimeEstimator(num_insts)

    output = []
    for line in line_itr:
        doc_id, html = parse_doc_jsonl_line(line)
        if num_insts:
            ticker.tick()
        try:
            extractor = Extractor(extractor='ArticleExtractor', html=html)
            core_text = extractor.getText()
        except Exception as e:
            print("Exception at Extractor")
            print(e)
            continue
        core_text = str(core_text)
        tt: TokenizedText = TokenizedText.from_text(core_text, tokenizer)
        output.append((doc_id, tt))
    return output

