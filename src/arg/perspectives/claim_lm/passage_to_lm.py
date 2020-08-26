from collections import Counter
from typing import List, Tuple

from arg.perspectives.load import get_perspective_dict
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import tokens_to_freq, smooth_ex, ClaimLM
from cache import load_from_pickle
from list_lib import flatten, left
from misc_lib import TimeEstimator


def passage_to_lm(tokenizer, claim, passages: List[Tuple[List[str], float]], alpha):
    claim_text = claim['text']
    claim_tokens = tokenizer.tokenize_stem(claim_text)

    tf = tokens_to_freq(flatten(left(passages)))
    c_tf = tokens_to_freq(claim_tokens)
    r_tf = smooth_ex(c_tf, tf, alpha)
    return r_tf


def get_valid_terms():
    perspective = get_perspective_dict()
    tokenizer = PCTokenizer()
    voca = set()
    for text in perspective.values():
        voca.update(tokenizer.tokenize_stem(text))
    return voca


def simplify_tf(tf, voca):
    c = Counter()
    for k, v in tf.items():
        if k in voca:
            c[k] = v
    return c


def get_train_passage_a_lms():
    data = load_from_pickle("pc_train_a_passages")
    entries, all_passages = data
    voca = get_valid_terms()

    tokenizer = PCTokenizer()
    bg_tf = tokens_to_freq(flatten(left(all_passages)))
    bg_tf = simplify_tf(bg_tf, voca)
    alpha = 0.99 # Smoothing with claim
    alpha2 = 0.3 # Smoothing with collection documents
    r = []
    ticker = TimeEstimator(len(entries))
    for c, passages in entries:
        r_tf = passage_to_lm(tokenizer, c, passages, alpha)
        r_tf = simplify_tf(r_tf, voca)
        c_tf = smooth_ex(r_tf, bg_tf, alpha2)
        r.append(ClaimLM(c['cId'], c['text'], c_tf))
        ticker.tick()

    return r
