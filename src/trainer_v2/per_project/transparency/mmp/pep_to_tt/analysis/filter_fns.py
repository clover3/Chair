
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from data_generator.tokenizer_wo_tf import get_tokenizer

g_bert_tokenizer = None

def contain_a_hat(term: str) -> bool:
    return "â" in term


def postfix_a(term: str) -> bool:
    global g_bert_tokenizer
    if g_bert_tokenizer is None:
        g_bert_tokenizer = get_tokenizer()

    tokens = g_bert_tokenizer.basic_tokenizer.tokenize(term)
    return "a" == "".join(tokens)[-1]


def is_number_like(term: str) -> bool:
    for i in range(10):
        if str(i) in term:
            return True

    return False


def or_combine(pattern_match: Callable[[str], bool]) -> Callable[[str, str], bool]:
    def match(q, d):
        return pattern_match(q) or pattern_match(d)

    return match


def and_combine(pattern_match: Callable[[str], bool]) -> Callable[[str, str], bool]:
    def match(q, d):
        return pattern_match(q) and pattern_match(d)

    return match


def get_qd_filter(filter_name) ->  Callable[[str, str], bool]:
    if filter_name[:4] == "not_":
        raw_fn = get_filter_inner(filter_name[4:])

        def fn(*args):
            return not raw_fn(*args)
        return fn
    else:
        return get_filter_inner(filter_name)


def get_score_filter(filter_name) -> Callable[[float], bool]:
    if filter_name.startswith("over_"):
        _, t = filter_name.split("_")
        t = float(t)

        def filter_fn(s: float) -> bool:
            return s > t

        return filter_fn
    else:
        raise ValueError


def get_filter(filter_name):
    if filter_name.startswith("over_"):
        s_filter = get_score_filter(filter_name)

        def filter_fn(qt, dt, s):
            return s_filter(float(s))

    else:
        qd_filter_fn = get_qd_filter(filter_name)

        def filter_fn(qt, dt, s):
            return qd_filter_fn(qt, dt)

    return filter_fn




def get_filter_inner(filter_name) ->  Callable[[str, str], bool]:
    return {
        'contain_a_hat': or_combine(contain_a_hat),
        'has_number': or_combine(is_number_like),
        'postfix_a': or_combine(postfix_a),
        "same": lambda qt, dt: qt == dt,

    }[filter_name]

