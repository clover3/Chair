import re
from typing import List, Tuple

import nltk

from arg.bm25 import BM25
from arg.counter_arg.header import Passage
from arg.counter_arg.methods.bm25_predictor import BasicTF
from arg.counter_arg.methods.tool import get_term_importance, sent_tokenize_newline
from bert_api.client_lib import BERTClient
from list_lib import lmap, lmap_pairing, left
from misc_lib import NamedNumber


def get_stance_check_candidate(text: str, bm25_module: BM25):
    sents = sent_tokenize_newline(text)
    term_importance = get_term_importance(bm25_module, sents)

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