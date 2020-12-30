from collections import Counter
from typing import Iterable, List

import spacy

from arg.bm25 import BM25Bare
from arg.counter_arg import header
from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import Passage
from arg.counter_arg.methods.bm25_predictor import BasicTF, remove_space_chars
from arg.query_weight.syntactic_weight_gen import QueryWeightGenBySyntactic
from cache import save_to_pickle, load_from_pickle
from list_lib import lmap
from misc_lib import TimeEstimator, NamedNumber
from tlm.retrieve_lm.stem import CacheStemmer


def count_df_spacy(passages: Iterable[Passage]) -> Counter:
    df = Counter()
    nlp_module = spacy.load("en_core_web_sm")
    stemmer = CacheStemmer()

    l_passage = list(passages)
    ticker = TimeEstimator(len(l_passage))
    for p in l_passage:
        ticker.tick()
        term_set = set()
        for token in nlp_module(p.text):
            try:
                term = stemmer.stem(token.text)
                term_set.add(term)
            except UnicodeDecodeError:
                pass

        for term in term_set:
            df[term] += 1

    return df


def build_df_spacy():
    for split in header.splits:
        df = count_df_spacy(enum_all_argument(split))
        save_to_pickle(df, "argu_{}_df_spacy".format(split))


def get_bm25_module_spacy(split):
    pickle_name = "argu_{}_df_spacy".format(split)
    df = load_from_pickle(pickle_name)
    N = {'training': 8148,
         'validation': 4074,
         'test': 4074,
         }[split]
    return BM25Bare(df, avdl=160, num_doc=N * 2, k1=1.2, k2=100, b=0.5)


def get_structured_scorer(split, query_reweight, body_weight):
    stemmer = CacheStemmer()
    bm25_module = get_bm25_module_spacy(split)
    query_gen = QueryWeightGenBySyntactic(query_reweight, stemmer)
    query_gen.load_nlp_cache_if_exist()
    basic_tf = BasicTF(query_gen.tokenize_stem)
    tf_cache = {}

    def get_tf_query(p: Passage) -> Counter:
        if p.id in tf_cache:
            return tf_cache[p.id]

        # lines = p.text.splitlines()
        # head = lines[0]
        # body = "\n".join(lines[1:])

        tf = Counter(query_gen.gen(p.text))
        # head_tf = query_gen.gen(head)
        #body_tf = Counter(query_gen.tokenize_stem(body))
        #
        # print("head: ", head_tf)
        # print("body: ", body_tf)

        #tf = counter_sum(head_tf, 1, body_tf, body_weight)
        tf_cache[p.id] = tf
        return tf

    def scorer(query_p: Passage, candidate: List[Passage]) -> List[NamedNumber]:
        q_tf = get_tf_query(query_p)
        q_tf = remove_space_chars(q_tf)

        def do_score(candidate_p: Passage) -> NamedNumber:
            if candidate_p.text == query_p.text:
                return NamedNumber(-99, "equal")
            p_tf = basic_tf.get_tf(candidate_p)
            return bm25_module.score_inner(q_tf, p_tf)

        scores = lmap(do_score, candidate)
        return scores

    return scorer


if __name__ == "__main__":
    build_df_spacy()