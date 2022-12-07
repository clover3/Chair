from collections import Counter
from typing import List, Iterable, Dict, Callable, Any

from adhoc.bm25_class import BM25
from arg.counter_arg import header
from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import ArguDataID, Passage
from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from cache import load_from_pickle, save_to_pickle
from list_lib import lmap
from misc_lib import NamedNumber


def counter_sum(counter1: Dict[Any, float], weight1: float,
                counter2: Dict[Any, float], weight2: float) -> Counter:

    new_counter = Counter()
    for k, v in counter1.items():
        new_counter[k] += v * weight1

    for k, v in counter2.items():
        new_counter[k] += v * weight2
    return new_counter


def get_scorer(split) -> Callable[[Passage, List[Passage]], List[NamedNumber]]:
    bm25_module = get_bm25_module(split)
    return get_scorer_from_bm25_module(bm25_module)


def remove_space_chars(d: Dict[str, float]) -> Counter:
    out_d = Counter()
    for k, v in d.items():
        if k.strip():
            out_d[k] = v

    return out_d


class BasicTF:
    def __init__(self, tokenize_stem):
        self.tokenize_stem = tokenize_stem
        self.tf_cache: Dict[ArguDataID, Counter] = {}

    def get_tf(self, p: Passage) -> Counter:
        if p.id in self.tf_cache:
            return self.tf_cache[p.id]

        text = p.text
        tokens = self.tokenize_stem(text)
        tf = Counter(tokens)
        tf = remove_space_chars(tf)
        self.tf_cache[p.id] = tf
        return tf


def get_scorer_from_bm25_module(bm25_module):
    basic_tf = BasicTF(bm25_module.tokenizer.tokenize_stem)

    def scorer(query_p: Passage, candidate: List[Passage]) -> List[NamedNumber]:
        q_tf = basic_tf.get_tf(query_p)

        def do_score(candidate_p: Passage) -> NamedNumber:
            if candidate_p.text == query_p.text:
                return NamedNumber(-99, "equal")
            p_tf = basic_tf.get_tf(candidate_p)
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
    return BM25(df, avdl=160, num_doc=N * 2, k1=0.1, k2=100, b=0.9)


def count_df(passages: Iterable[Passage]) -> Counter:
    tokenizer = KrovetzNLTKTokenizer()
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