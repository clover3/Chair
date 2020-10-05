import copy
from collections import Counter

import math

from list_lib import dict_value_map


def get_lm_log(lm: Counter) -> Counter:
    return Counter(dict_value_map(math.log, lm))


def subtract(counter1: Counter, counter2: Counter) -> Counter:
    output = copy.deepcopy(counter1)
    output.subtract(counter2)
    return output


def least_common(counter: Counter, n):
    l = list(counter.items())
    l.sort(key=lambda x: x[1])
    for e in l[:n]:
        yield e


def smooth(target_lm: Counter, bg_lm: Counter, alpha):
    output = Counter()
    for k, v in bg_lm.items():
        output[k] = target_lm[k] * (1-alpha) + alpha * v

    return output


def smooth_ex(target_lm: Counter, bg_lm: Counter, alpha):
    output = Counter()
    keys = set(bg_lm.keys())
    keys.update(target_lm.keys())
    for k in keys:
        v = bg_lm[k]
        output[k] = target_lm[k] * (1-alpha) + alpha * v

    return output


def get_log_odd(topic_lm, bg_lm, alpha):
    log_topic_lm = get_lm_log(smooth(topic_lm.LM, bg_lm, alpha))
    log_bg_lm = get_lm_log(bg_lm)
    log_odd: Counter = subtract(log_topic_lm, log_bg_lm)
    return log_odd