from collections import Counter

import math

from cache import save_to_pickle, load_from_pickle
from models.classic.stopword import load_stopwords
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic


def get_all_term_odd(tf_c, tf_nc, smoothing=0.9):
    if tf_c == 0 and tf_nc == 0:
        return 0
    c_ctf = sum(tf_c.values())
    nc_ctf = sum(tf_nc.values())

    odd_dict = Counter()
    for key in tf_c:
        P_w_C = tf_c[key] / c_ctf
        P_w_N = tf_nc[key] / nc_ctf

        logC = math.log(P_w_C * smoothing + P_w_N * (1 - smoothing))
        logNC = math.log(P_w_C * (1 - smoothing) + P_w_N * smoothing)
        assert (math.isnan(logC) == False)
        assert (math.isnan(logNC) == False)
        log_odd = logC - logNC
        odd_dict[key] = log_odd
    return odd_dict


def build_lm_from_tokens_list(doc) -> Counter:
    tf = Counter()
    for segment in doc:
        tf.update(segment)
    return tf


def work():
    topic = "abortion"
    c_tf, nc_tf = count_controversy(topic)

    save_to_pickle((c_tf, nc_tf), "abortion_clm")
    display(c_tf, nc_tf)

def start_from_pickle():
    c_tf, nc_tf = load_from_pickle("abortion_clm")
    display(c_tf, nc_tf)


def display(c_tf, nc_tf):
    odd_dict = get_all_term_odd(c_tf, nc_tf, 0.95)

    def contrib(e):
        key, value = e
        return (c_tf[key] + nc_tf[key]) * value

    odd_list = list(odd_dict.items())
    odd_list.sort(key=contrib, reverse=True)
    stopword = load_stopwords()

    def valid(e):
        key, value = e
        return key not in stopword and c_tf[key] > 10 and nc_tf[key] > 10

    acc = 0
    for key, value in odd_list:
        acc += value * (c_tf[key] + nc_tf[key])

    ctf = sum(c_tf.values())+sum(nc_tf.values())
    print(acc, acc/ctf)

    odd_list = list(filter(valid, odd_list))
    print("Top Controversial ")
    for key, value in odd_list[:30]:
        print(key, c_tf[key], nc_tf[key], odd_dict[key])
    print("Least Controversial ")
    for idx in range(len(odd_list) - 1, len(odd_list) - 1 - 20, -1):
        key, value = odd_list[idx]
        print(key, contrib(odd_list[idx]), c_tf[key], nc_tf[key], odd_dict[key])


def count_controversy(topic):
    token_controversial = ["controversy", "controversial"]

    def contain_controversy(tf):
        for t in token_controversial:
            if t in tf:
                return True
        return False

    tokens_dict = ukp_load_tokens_for_topic(topic)
    c_tf = Counter()
    nc_tf = Counter()
    for doc_id, doc in tokens_dict.items():
        tf = build_lm_from_tokens_list(doc)
        if contain_controversy(tf):
            c_tf.update(tf)
        else:
            nc_tf.update(tf)
    return c_tf, nc_tf


if __name__ == "__main__":
    start_from_pickle()