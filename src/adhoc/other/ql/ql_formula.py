import math
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def apply_smoothing(q_term_bg_prob, dl, mu, p_w_d_from_d):
    p_w_d_from_d = (dl / (dl + mu)) * p_w_d_from_d
    p_w_d_bg = (mu / (dl + mu)) * q_term_bg_prob
    p_w_d = p_w_d_from_d + p_w_d_bg
    return p_w_d


def apply_jm_smoothing(q_term_bg_prob, dl, _mu, p_w_d_from_d):
    weight = 0.35
    p_w_d_from_d = (1 - weight) * p_w_d_from_d
    p_w_d_bg = weight * q_term_bg_prob
    p_w_d = p_w_d_from_d + p_w_d_bg
    return p_w_d


def ql_scoring_inner(tf, qf, dl, bg_prob_val, mu):
    p_w_d_from_d = tf / dl  # P(w|d)
    p_w_d = apply_smoothing(bg_prob_val, dl, mu, p_w_d_from_d)
    log_p_w_d = math.log(p_w_d)
    # per_q_term_score = qf * log_p_w_d
    per_q_term_score = log_p_w_d
    return per_q_term_score


def ql_scoring_from_p_qf(p_w_d_from_d, dl, bg_prob_val, mu):
    p_w_d = apply_smoothing(bg_prob_val, dl, mu, p_w_d_from_d)
    log_p_w_d = math.log(p_w_d)
    per_q_term_score = log_p_w_d
    return per_q_term_score


#
def ql_translate_scoring(
        bg_prob: Callable[[str], float],
        translate: Callable[[str, str], float],
        q_tokens: list[str], d_tokens: list[str]):
    mu = 2000
    q_counter = Counter(q_tokens)
    d_counter = Counter(d_tokens)
    dl = sum(d_counter.values())
    score = 0
    for q_term, q_term_f in q_counter.items():
        # w = d_term
        p_w_d_from_d = 0  # P(w|d)
        for d_term, d_term_tf in d_counter.items():
            p_d_term_d = d_term_tf / dl
            p_w_d_from_d += translate(q_term, d_term) * p_d_term_d

        p_w_d_from_d = (dl / (dl + mu)) * p_w_d_from_d
        p_w_d_bg = (mu / (dl + mu)) * bg_prob(q_term)
        p_w_d = p_w_d_from_d + p_w_d_bg
        log_p_w_d = math.log(p_w_d)
        score += q_term_f * log_p_w_d
    return score


def ql_scoring(
        bg_prob: Callable[[str], float],
        q_tokens: list[str], d_tokens: list[str]):
    mu = 2000

    q_counter = Counter(q_tokens)
    d_counter = Counter(d_tokens)
    dl = sum(d_counter.values())
    score = 0
    for q_term, q_term_f in q_counter.items():
        # w = d_term
        tf = d_counter[q_term]
        bg_prob_val = bg_prob(q_term)
        per_q_term_score = ql_scoring_inner(tf, q_term_f, dl, bg_prob_val, mu)

        score += per_q_term_score
    return score

def main():
    def translate(_q_term, _d_term):
        return 0

    def bg_prob(q_term):
        return 1 / 300

    query = "book price"
    doc = "The price for this book is not hundress dollars"
    q_tokens = query.split()
    d_tokens = doc.split()
    ret = ql_translate_scoring(bg_prob, translate, q_tokens, d_tokens)
    print(ret)


if __name__ == "__main__":
    main()