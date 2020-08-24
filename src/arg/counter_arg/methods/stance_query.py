import re
from collections import Counter
from typing import List, Tuple

import nltk
from nltk import sent_tokenize

from arg.counter_arg.header import Passage
from arg.counter_arg.methods.bm25_predictor import BasicTF
from arg.perspectives.collection_based_classifier import NamedNumber
from bert_api.client_lib import BERTClient
from list_lib import lmap, flatten, lmap_pairing, left
from models.classic.bm25 import BM25


def sent_tokenize_newline(text):
    sents = sent_tokenize(text)
    r = []
    for s in sents:
        for new_sent in s.split("\n"):
            r.append(new_sent)
    return r


def get_stance_check_candidate(text: str, bm25_module: BM25):
    sents = sent_tokenize_newline(text)
    tokens = flatten([bm25_module.tokenizer.tokenize_stem(s) for s in sents])
    q_tf = Counter(tokens)
    term_importance = Counter()
    for term, tf in q_tf.items():
        term_importance[term] += bm25_module.term_idf_factor(term) * tf

    def is_heading_num(s):
        return re.match(r'^\[(\d{1,3}|i{1,5})\]', s) is not None

    r = []
    for sent in sents:
        if not sent.strip():
            continue

        if is_heading_num(sent.strip()):
            continue
        tokens = nltk.tokenize.word_tokenize(sent)
        tokens = set(tokens)

        def per_token_score(t):
            s = bm25_module.tokenizer.stemmer.stem(t)
            return term_importance[s]

        scores: List[Tuple[str, float]] = lmap_pairing(per_token_score, tokens)
        scores.sort(key=lambda x: x[1], reverse=True)
        terms = left(scores[:5])

        candidate = sent, terms
        r.append(candidate)
    return r


def issue_query(client: BERTClient, raw_payload: List[Tuple[str, str]]):
    payload = []
    for sent, term in raw_payload:
        payload.append((term, sent))

    return client.request_multiple(payload)



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