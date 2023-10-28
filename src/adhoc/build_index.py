import os
import pathlib
import pickle
from collections import Counter
from typing import Dict, List, Tuple
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import TELI
from trainer_v2.chair_logging import c_log


def count_df_from_tokenized(
        passages: Iterable[Tuple[str, List[str]]]) -> Counter:
    df = Counter()
    for doc_id, tokens in passages:
        for term in set(tokens):
            df[term] += 1

    return df


def count_dl_from_tokenized(
        passages: Iterable[Tuple[str, List[str]]]) -> Dict:
    dl = {}
    for doc_id, tokens in passages:
        dl[doc_id] = len(tokens)
    return dl


def build_inverted_index(
        tokenized_itr: Iterable[Tuple[str, List[str]]],
        ignore_voca,
        num_docs=None,
        term_df_cut_to_discard=1000000,
        term_df_cut_to_warn=10000):
    print("ignore voca", ignore_voca)
    if num_docs is not None:
        itr = TELI(tokenized_itr, num_docs)
    else:
        itr = tokenized_itr
    max_l_posting = 0
    inverted_index: Dict[str, List[Tuple[str, int]]] = {}
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            if token in ignore_voca:
                continue
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((doc_id, cnt))
            if len(inverted_index[token]) > term_df_cut_to_discard:
                print("Discard term {}".format(token))
                inverted_index[token] = []
                ignore_voca.add(token)

            l_posting = len(inverted_index[token])
            if l_posting - max_l_posting > term_df_cut_to_warn:
                print("{} has {} items".format(token, l_posting))
                max_l_posting = l_posting
    return inverted_index


def build_inverted_index_plus(
        tokenized_itr: Iterable[Tuple[str, List[str]]],
        ignore_voca,
        num_docs=None,
        term_df_cut_to_discard=1000000,
        term_df_cut_to_warn=10000):
    print("ignore voca", ignore_voca)
    if num_docs is not None:
        itr = TELI(tokenized_itr, num_docs)
    else:
        itr = tokenized_itr
    max_l_posting = 0
    inverted_index: Dict[str, List[Tuple[str, int]]] = {}
    df = Counter()
    dl = {}
    cdf = 0
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            if token in ignore_voca:
                continue
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((doc_id, cnt))
            if len(inverted_index[token]) > term_df_cut_to_discard:
                print("Discard term {}".format(token))
                inverted_index[token] = []
                ignore_voca.add(token)

            l_posting = len(inverted_index[token])
            if l_posting - max_l_posting > term_df_cut_to_warn:
                print("{} has {} items".format(token, l_posting))
                max_l_posting = l_posting

            df[token] += 1
        dl[doc_id] = sum(count.values())
        cdf += 1
    return {
        'df': df,
        "dl": dl,
        "cdf": cdf,
        'inverted_index': inverted_index,
    }


def count_dl_df(
        tokenized_itr: Iterable[Tuple[str, List[str]]],
        num_docs=None):
    if num_docs is not None:
        itr = TELI(tokenized_itr, num_docs)
    else:
        itr = tokenized_itr
    df = Counter()
    dl = {}
    cdf = 0
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            df[token] += 1
        dl[doc_id] = sum(count.values())
        cdf += 1
    return {
        'df': df,
        "dl": dl,
        "cdf": cdf,
    }


def save_inv_index_to_pickle(conf, outputs):
    inverted_index = outputs["inverted_index"]
    dl = outputs["dl"]
    df = outputs["df"]
    c_log.info("Saving df")
    dir_maybe = os.path.dirname(conf.df_path)
    pathlib.Path(dir_maybe).mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(conf.df_path, "wb"))
    c_log.info("Saving dl")
    pickle.dump(dl, open(conf.dl_path, "wb"))
    c_log.info("Saving inv_index")
    pickle.dump(inverted_index, open(conf.inv_index_path, "wb"))
    avdl = sum(dl.values()) / len(dl)
    c_log.info("Avdl: %d", avdl)
