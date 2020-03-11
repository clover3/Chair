from collections import Counter

import math

from arg.claim_building.count_ngram import merge_subword
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


def build_uni_lm_from_tokens_list(doc) -> Counter:
    tf = Counter()
    for segment in doc:
        tf.update(segment)
    return tf


def work():
    topic = "abortion"
    c_tf, nc_tf = count_controversy(topic)

    save_to_pickle((c_tf, nc_tf), "abortion_clm")
    display(c_tf, nc_tf, "controversial", "non-controversial")

def start_from_pickle():
    c_tf, nc_tf = load_from_pickle("abortion_clm")
    display(c_tf, nc_tf, "controversial", "non-controversial")


def display(tf1, tf2, label_name1="pos", label_name2="neg"):
    odd_dict = get_all_term_odd(tf1, tf2, 0.95)

    def contrib(e):
        key, value = e
        return (tf1[key] + tf2[key]) * value

    odd_list = list(odd_dict.items())
    odd_list.sort(key=contrib, reverse=True)
    stopword = load_stopwords()

    def valid(e):
        key, value = e
        return key not in stopword and tf1[key] > 10 and tf2[key] > 10

    acc = 0
    for key, value in odd_list:
        acc += value * (tf1[key] + tf2[key])

    ctf = sum(tf1.values()) + sum(tf2.values())
    print(acc, acc/ctf)

    k = 50

    odd_list = list(filter(valid, odd_list))
    print("Top {} ".format(label_name1))
    for key, value in odd_list[:k]:
        print(key, tf1[key], tf2[key], odd_dict[key])
    print("Top {} ".format(label_name2))
    for idx in range(len(odd_list) - 1, len(odd_list) - 1 - k, -1):
        key, value = odd_list[idx]
        print(key, contrib(odd_list[idx]), tf1[key], tf2[key], odd_dict[key])


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
        doc = [merge_subword(s) for s in doc]
        tf = build_uni_lm_from_tokens_list(doc)
        if contain_controversy(tf):
            c_tf.update(tf)
        else:
            nc_tf.update(tf)
    return c_tf, nc_tf


if __name__ == "__main__":
    work()