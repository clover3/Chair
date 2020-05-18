import string
from typing import List, Iterable, Callable, Dict, Tuple, Set

import math
import nltk

from arg.pf_common.text_processing import re_tokenize
from list_lib import lmap, lmap_pairing, lfilter, left, flatten
from misc_lib import TimeEstimator


def remove_duplicate(doc_list: List[List[str]]):
    def para_hash(doc: List[str]):
        return " ".join(doc)

    hash_set = set()

    for doc in doc_list:
        hash = para_hash(doc)
        if hash in hash_set:
            continue

        hash_set.add(hash)
        yield doc


def enum_paragraph(docs: List[List[str]]) -> Iterable[List[str]]:
    step_size = 100
    window_size = 300

    for doc in remove_duplicate(docs):
        st = 0
        while st < len(doc):
            para: List[str] = doc[st:st + window_size]
            yield para
            st += step_size


def paragraph_scorer(idf_fn: Callable[[str], float],
                     q_terms: Set[str],
                     paragraph: List[str]) -> float:
    paragraph_terms = set(paragraph)
    mentioned_terms = lfilter(lambda x: x in paragraph_terms, q_terms)
    mentioned_terms = re_tokenize(mentioned_terms)

    score = sum(lmap(idf_fn, mentioned_terms))
    return score


def select_paragraph(docs: Dict[str, List[List[str]]],
                     clue12_13_df,
                     claim_list: List[Dict],
                     strategy="topk",
                     ) -> List[Tuple[str, List[List[str]]]]:

    claim_id_to_text: Dict[int, str] = {c['cId']: c['text'] for c in claim_list}

    cdf = 50 * 1000 * 1000
    top_k = 100
    not_found_set = set()

    def idf(term: str):
        if term not in clue12_13_df:
            if term in string.printable:
                return 0
            not_found_set.add(term)

        return math.log((cdf + 0.5) / (clue12_13_df[term] + 0.5))

    r: List[Tuple[str, List[List[str]]]] = []
    ticker = TimeEstimator(len(docs))
    for claim_id, docs in docs.items():
        claim_text = claim_id_to_text[int(claim_id)]
        q_terms = set(re_tokenize(nltk.tokenize.word_tokenize(claim_text)))

        def scorer(para: List[str]) -> float:
            return paragraph_scorer(idf, q_terms, para)

        max_score = sum(lmap(idf, q_terms))

        def get_best_per_doc(doc: List[str]) -> List[Tuple[List[str], float]]:
            paragraph_list: Iterable[List[str]] = enum_paragraph([doc])
            paragraph_scored_list: List[Tuple[List[str], float]] = lmap_pairing(scorer, paragraph_list)
            paragraph_scored_list.sort(key=lambda x: x[1], reverse=True)
            return paragraph_scored_list[:1]

        selected: List[Tuple[List[str], float]] = list(flatten(lmap(get_best_per_doc, docs)))

        # if strategy == "topk":
        #     selected: List[Tuple[List[str], float]] = paragraph_scored_list[:top_k]
        # elif strategy == "cutoff":
        #     cut_off = max_score * 0.6
        #     selected: List[Tuple[List[str], float]] = lfilter(lambda x: x[1] > cut_off, paragraph_scored_list)
        # else:
        #     assert False

        e = claim_id, left(selected)
        r.append(e)
        ticker.tick()

    return r
