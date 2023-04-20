from collections import defaultdict
from math import log
from typing import Iterable

from krovetzstemmer import Stemmer

from adhoc.bm25_class import BM25
from cpath import output_path
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bm25t import GlobalAlign, BM25T


def get_mmp_bm25():
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, k1=0.1, k2=0, b=0.1)
    return bm25


def load_align_weights() -> Iterable[GlobalAlign]:
    global_align_path = path_join(
        output_path, "msmarco", "passage", "when_local_avg_align")
    itr = tsv_iter(global_align_path)

    def parse_row(row):
        return GlobalAlign(
            int(row[0]),
            row[1],
            float(row[2]), int(row[3]), int(row[4]))

    return map(parse_row, itr)


def build_table():
    stemmer = Stemmer()
    min_tf = 0
    out_mapping = {}
    n_all = 0
    for t in load_align_weights():
        n_all += 1
        rate = t.n_pos_appear / t.n_appear
        if t.n_appear >= min_tf and t.score > 0.01 and rate > 0.6:
            word = stemmer(t.word)
            out_mapping[word] = t.score
    print("Selected {} from {} items".format(len(out_mapping), n_all))
    return out_mapping


def build_table2():
    stemmer = Stemmer()
    min_tf = 0
    out_mapping = {}
    n_all = 0
    max_freq = 96135

    for t in load_align_weights():
        n_all += 1
        if t.n_appear >= min_tf:
            n_neg = t.n_appear - t.n_pos_appear
            log_pos = log(t.n_pos_appear + 1 / max_freq)
            log_neg = log(n_neg + 1 / max_freq)
            log_odd = log_pos - log_neg
            if log_odd > 0.01:
                weight = log_odd / 5
                word = stemmer(t.word)
                out_mapping[word] = t.score
    print("Selected {} from {} items".format(len(out_mapping), n_all))
    return out_mapping


def load_trained():
    save_path = path_join(
        output_path, "msmarco", "passage", "when_trained")
    out_mapping = {}
    for k, v in tsv_iter(save_path):
        out_mapping[k] = float(v)
    return out_mapping


def get_bm25t_when():
    mapping = defaultdict(dict)
    mapping['when'] = build_table()
    bm25 = get_mmp_bm25()
    bm25t = BM25T(mapping, bm25.core)
    # given a raw word, if raw word exactly matches
    return bm25t


def get_bm25t_when2():
    mapping = defaultdict(dict)
    mapping['when'] = build_table2()
    bm25 = get_mmp_bm25()
    bm25t = BM25T(mapping, bm25.core)
    # given a raw word, if raw word exactly matches
    return bm25t


def get_bm25t_when_trained():
    mapping = defaultdict(dict)
    mapping['when'] = load_trained()
    bm25 = get_mmp_bm25()
    bm25t = BM25T(mapping, bm25.core)
    # given a raw word, if raw word exactly matches
    return bm25t


def get_candidate_voca():
    itr = load_align_weights()
    stemmer = Stemmer()
    min_tf = 10
    l = []
    for t in itr:
        if t.n_appear >= min_tf and t.score > 0.01:
            word = stemmer(t.word)
            l.append(word)
    print("Selected {}".format(len(l)))
    voca = {}
    for idx, term in enumerate(l):
        voca[term] = idx+1
    return voca