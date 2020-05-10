import os
import pickle
from collections import Counter
from typing import List

import nltk

from arg.perspectives.basic_analysis import PerspectiveCandidate
from arg.perspectives.n_gram_feature_collector import PCNGramFeature
from arg.perspectives.types import CPIDPair
from data_generator.job_runner import sydney_working_dir
from data_generator.subword_convertor import SubwordConvertor
from misc_lib import exist_or_mkdir
from misc_lib import group_by


def to_regular_words(tokenizer, paragraph):
    return NotImplemented


def group_and_extract_ngram(subword_convertor: SubwordConvertor,
                            max_para, entries) -> List[PCNGramFeature]:
    n_list = [1, 2, 3]
    all_n_grams = {n: Counter() for n in n_list}

    # output_entry = cpid, label, paragraph, c_score, p_score
    def get_cpid(entry) -> CPIDPair:
        return entry[0]

    def get_paragraph(entry):
        return entry[2]

    def get_label(entry):
        return entry[1]

    grouped = group_by(entries, get_cpid)

    r = []
    for cpid, entries in grouped.items():
        for e in entries[:max_para]:
            word_list = subword_convertor.get_words(get_paragraph(e))
            for n in n_list:
                ngram_list = nltk.ngrams(word_list, n)
                all_n_grams[n].update(ngram_list)

        cid, pid = cpid
        claim_pers = PerspectiveCandidate(
            label=str(get_label(entries[0])),
            cid=cid,
            pid=pid,
            claim_text="",
            p_text=""
        )
        pc_ngram_feature = PCNGramFeature(claim_pers=claim_pers, n_grams=all_n_grams)
        r.append(pc_ngram_feature)
    return r


class PCNgramSubwordWorker:
    def __init__(self, max_para, input_job_name, out_dir):
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)
        self.max_para = max_para
        self.subword_convertor = SubwordConvertor()

    def work(self, job_id):
        entries: List = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))
        all_result: List[PCNGramFeature] = group_and_extract_ngram(self.subword_convertor, self.max_para, entries)
        pickle.dump(all_result, open(os.path.join(self.out_dir, str(job_id)), "wb"))

