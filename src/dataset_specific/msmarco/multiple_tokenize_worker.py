import os
import pickle
from typing import List

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import MSMarcoDataReader, MSMarcoDoc, load_per_query_docs
from list_lib import lmap, drop_empty_elem
from log_lib import log_variables
from misc_lib import TimeEstimator, exist_or_mkdir
from nltk import sent_tokenize


def crop_to_space(text, max_char):
    if len(text) <= max_char:
        return text
    last_space = -1
    for i in range(max_char):
        if text[i] == ' ':
            last_space = i


    if last_space == -1:
        return text[:max_char]
    else:
        return text[:last_space]


def split_by_space(text, max_sent_length):
    i = 0
    st = 0
    last_space = 0
    while i < len(text):
        if i == ' ':
            last_space = i
        if i - st >= max_sent_length:
            if last_space > st:
                new_piece = text[st:last_space]
                st = i + 1
            else:
                new_piece = text[st:i]
                st = i

            yield new_piece
        i += 1
    if i > st:
        yield text[st:]


class MultipleTokenizeWorker:
    def __init__(self,
                 split,
                 query_group,
                 candidate_docs_d,
                 max_sent_length,
                 max_title_length,
                 out_dir):
        self.query_group = query_group
        self.candidate_docs_d = candidate_docs_d
        self.out_dir = out_dir
        self.bert_tokenizer = get_tokenizer()
        self.stem_tokenizer = KrovetzNLTKTokenizer()
        self.max_sent_length = max_sent_length
        self.max_title_length = max_title_length
        self.ms_reader = MSMarcoDataReader(split)
        self.text_dir_name = 'text'
        self.bert_tokens_dir_name = 'bert_tokens'
        self.stemmed_tokens_dir_name = 'stemmed_tokens'

        for name in [self.text_dir_name, self.bert_tokens_dir_name, self.stemmed_tokens_dir_name]:
            exist_or_mkdir(os.path.join(self.out_dir, name))

    def work(self, job_id):
        qid_list = self.query_group[job_id]
        ticker = TimeEstimator(len(qid_list))
        missing_rel_cnt = 0
        missing_nrel_cnt = 0
        def empty_doc_fn(query_id, doc_id):
            rel_docs = self.ms_reader.qrel[query_id]
            nonlocal missing_rel_cnt
            nonlocal missing_nrel_cnt
            if doc_id in rel_docs:
                missing_rel_cnt += 1
            else:
                missing_nrel_cnt += 1

        for qid in qid_list:
            if qid not in self.candidate_docs_d:
                continue

            docs: List[MSMarcoDoc] = load_per_query_docs(qid, empty_doc_fn)
            ticker.tick()

            target_docs = self.candidate_docs_d[qid]
            text_d = {}
            bert_tokens_d = {}
            stemmed_tokens_d = {}

            for d in docs:
                if d.doc_id in target_docs:
                    title = d.title
                    title = crop_to_space(title, self.max_title_length)

                    body_sents = sent_tokenize(d.body)
                    new_body_sents = self.resplit_body_sents(body_sents)
                    text_d[d.doc_id] = title, new_body_sents

                    for tokenize_fn, save_dict in [(self.bert_tokenizer.tokenize, bert_tokens_d),
                                                 (self.stem_tokenizer.tokenize_stem, stemmed_tokens_d)]:
                        title_tokens = tokenize_fn(title)
                        body_tokens_list = lmap(tokenize_fn, new_body_sents)
                        save_dict[d.doc_id] = (title_tokens, body_tokens_list)

            todo = [
                (text_d, self.text_dir_name),
                (bert_tokens_d, self.bert_tokens_dir_name),
                (stemmed_tokens_d, self.stemmed_tokens_dir_name),
            ]

            for tokens_d, dir_name in todo:
                save_path = os.path.join(self.out_dir, dir_name, str(qid))
                pickle.dump(tokens_d, open(save_path, "wb"))

    def resplit_body_sents(self, body_sents):
        new_body_sents = []
        for sent in body_sents:
            if len(sent) > self.max_sent_length:
                new_sents = split_by_space(sent, self.max_sent_length)
                new_body_sents.extend(new_sents)
            else:
                new_body_sents.append(sent)
        return new_body_sents
