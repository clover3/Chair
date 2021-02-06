import os
from collections import Counter
from typing import List, Tuple

import numpy as np
from krovetzstemmer import Stemmer
from lime import lime_text
from math import log

from cpath import data_path
from explain.genex.load import QueryDoc, parse_problem
from galagos.parse import load_df
from list_lib import lmap
from misc_lib import TimeEstimator, get_second
from tlm.retrieve_lm.stem import CacheStemmer


def load_df_stemmed(term_stat_path):
    stemmer = Stemmer()
    df = load_df(term_stat_path)

    new_df = Counter()
    for key, value in df.items():
        try:
            new_df[stemmer.stem(key)] += value
        except UnicodeDecodeError:
            pass
    return new_df


def load_df_for(data_name):
    if data_name == "wiki":
        return load_df_stemmed(os.path.join(data_path, "enwiki", "tf_stat"))
    elif data_name == "clue":
        return load_df_stemmed(os.path.join(data_path, "clueweb12_B13_termstat.txt"))
    else:
        assert False


def load_idf_fn_for(data_name):
    df_dict = load_df_for(data_name)

    cdf = max(df_dict.values()) * 10

    def get_idf(term):
        idf = log(1 + cdf / (df_dict[term]+1))
        return idf
    return get_idf


def explain_by_lime_idf(data: List[str], get_idf) -> List[Tuple[str, float]]:
    stemmer = CacheStemmer()

    def split(t):
        return t.split()

    explainer = lime_text.LimeTextExplainer(split_expression=split, bow=True)

    def evaluate_score(problems: List[str]):
        scores = []
        for problem in problems:
            score = solve(problem)
            scores.append([0, score])
        return np.array(scores)

    def solve(problem: str):
        tokens = split(problem)
        if "[SEP]" not in tokens:
            return 0
        e: QueryDoc = parse_problem(tokens)
        q_terms = lmap(stemmer.stem, e.query)
        doc_terms = lmap(stemmer.stem, e.doc)
        tf = Counter(doc_terms)
        q_terms_set = set(q_terms)
        score = 0
        for term, cnt in tf.items():
            if term in q_terms_set:
                idf = get_idf(term)
                idf = 1
                score += log(1+cnt) * idf
            # TODO add idf multiplication
        return score

    explains = []
    tick = TimeEstimator(len(data))
    for entry in data:
        assert type(entry) == str
        exp = explainer.explain_instance(entry, evaluate_score, num_features=512)
        # l = list(exp.local_exp[1])
        # l.sort(key=get_first)
        # indices, scores = zip(*l)
        l2 = exp.as_list()
        l2.sort(key=get_second, reverse=True)
        explains.append(l2)
        tick.tick()
    return explains

