from collections import Counter
from typing import List, Iterable, Dict

from arg.counter_arg import header
from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import ArguDataID, Passage
from arg.perspectives.collection_based_classifier import NamedNumber
from arg.perspectives.pc_tokenizer import PCTokenizer
from cache import load_from_pickle, save_to_pickle
from list_lib import lmap
from models.classic.bm25 import BM25


def get_scorer(split):
    bm25_module = get_bm25_module(split)
    return get_scorer_from_bm25_module(bm25_module)


def get_scorer_from_bm25_module(bm25_module):
    tf_cache: Dict[ArguDataID, Counter] = {}

    tf_cache: Dict[ArguDataID, Counter] = {}

    def get_tf(p: Passage) -> Counter:
        if p.id in tf_cache:
            return tf_cache[p.id]

        tokens = bm25_module.tokenizer.tokenize_stem(p.text)
        tf = Counter(tokens)
        tf_cache[p.id] = tf
        return tf

    def scorer(query_p: Passage, candidate: List[Passage]) -> List[NamedNumber]:
        q_tf = get_tf(query_p)

        def do_score(candidate_p: Passage) -> NamedNumber:
            if candidate_p.text == query_p.text:
                return NamedNumber(-99, "equal")
            p_tf = get_tf(candidate_p)
            return bm25_module.score_inner(q_tf, p_tf)

        scores = lmap(do_score, candidate)
        return scores

    return scorer


def get_bm25_module(split):
    pickle_name = "argu_{}_df".format(split)
    df = load_from_pickle(pickle_name)
    N = {'training': 8148,
         'validation': 4074,
         'test': 4074,
         }[split]
    return BM25(df, avdl=160, num_doc=N * 4, k1=1.2, k2=100, b=0.5)


def count_df(passages: Iterable[Passage]) -> Counter:
    tokenizer = PCTokenizer()
    df = Counter()
    for p in passages:
        tokens = tokenizer.tokenize_stem(p.text)

        for term in set(tokens):
            df[term] += 1

    return df


def build_df():
    for split in header.splits:
        df = count_df(enum_all_argument(split))
        save_to_pickle(df, "argu_{}_df".format(split))


if __name__ == "__main__":
    build_df()