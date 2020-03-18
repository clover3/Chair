import collections
import os
import string
from typing import Iterable, Counter

import math
import nltk
import numpy as np
from numpy import std, median

from arg.claim_building.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.clueweb_helper import preload_tf, load_tf
from arg.perspectives.pc_run_path import train_query_indices, ranked_list_save_root
from cache import load_from_pickle
from galagos.basic import merge_ranked_list_list, load_galago_ranked_list
from list_lib import lmap, lmap_w_exception, l_to_map, dict_map, lfilter, left, right
from misc_lib import average
from models.classic.lm_counter_io import LMClassifier
from sydney_clueweb.clue_path import index_name_list


def dirichlet_smoothing(tf, dl, c_tf, c_ctf):
    mu = 1500
    denom = tf + mu * (c_tf / c_ctf)
    nom = dl + mu
    return denom / nom


class CollectionInterface:
    def __init__(self, do_lowercase=True):
        print("CollectionInterface __init__")
        self.available_disk_list = index_name_list[:1]
        self.ranked_list = l_to_map(load_all_ranked_list, self.available_disk_list)
        self.do_lowercase = do_lowercase
        self.collection_tf = load_from_pickle("collection_tf")
        self.collection_ctf = sum(self.collection_tf.values())
        print("CollectionInterface __init__ Done")

    def get_ranked_documents_tf(self, claim_id, perspective_id, allow_not_found=False) -> Iterable[Counter]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        ranked_list = self.get_ranked_list(query_id)
        doc_ids = lmap(lambda x: x[0], ranked_list)
        preload_tf(doc_ids)

        def do_load_tf(doc_id):
            try:
                counter = load_tf(doc_id)
                if self.do_lowercase:
                    counter = {key.lower(): value for key, value in counter.items()}
            except KeyError:
                if allow_not_found:
                    counter = None
                else:
                    raise
            return counter

        return lmap(do_load_tf, doc_ids)

    def get_ranked_list(self, query_id):
        ranked_list_list = []
        last_error = None
        for disk_name in self.available_disk_list:
            try:
                ranked_list = self.ranked_list[disk_name][query_id]
                ranked_list_list.append(ranked_list)
            except KeyError as e:
                print(e)
                last_error = e
                pass
        if not ranked_list_list:
            raise last_error

        return merge_ranked_list_list(ranked_list_list)

    def tf_collection(self, term):
        return self.collection_tf[term], self.collection_ctf


def div_by_doc_len(doc: Counter):
    doc_len = sum(doc.values())
    if doc_len == 0:
        return doc
    else:
        c = collections.Counter()
        for term, count in doc.items():
            c[term] = count / doc_len
        return c


def get_feature_binary_model(claim_id,
                             perspective_id,
                             claim_text,
                             perspective_text,
                             collection_interface: CollectionInterface,
                             is_mention_fn,
                             ):

    def is_mention(doc):
        return is_mention_fn(doc, claim_text, perspective_text)

    ranked_docs = collection_interface.get_ranked_documents_tf(claim_id, perspective_id)
    mentioned_docs = lfilter(is_mention, ranked_docs)

    docs_rel_freq = lmap(div_by_doc_len, mentioned_docs)
    num_doc = len(mentioned_docs)
    p_w_m = average_tf_over_docs(docs_rel_freq, num_doc)

    return p_w_m, num_doc


def average_tf_over_docs(docs_rel_freq, num_doc):
    p_w_m = collections.Counter()
    for doc in docs_rel_freq:
        for term, freq in doc.items():
            p_w_m[term] += freq / num_doc
    return p_w_m


def get_feature_weighted_model(claim_id,
                               perspective_id,
                               claim_text,
                               perspective_text,
                               collection_interface: CollectionInterface,
                               ):
    ranked_docs = collection_interface.get_ranked_documents_tf(claim_id, perspective_id)
    cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
    cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
    cp_tokens_count = collections.Counter(cp_tokens)

    def get_mention_prob(doc):
        dl = sum(doc.values())
        log_prob = 0

        for term, cnt in cp_tokens_count.items():
            c_tf, c_ctf = collection_interface.tf_collection(term)
            p_w = dirichlet_smoothing(cnt, dl, c_tf, c_ctf)
            log_prob += math.log(p_w)

        return math.exp(log_prob)

    docs_rel_freq = lmap(div_by_doc_len, ranked_docs)
    p_M_bar_D_list = lmap(get_mention_prob, docs_rel_freq)

    def apply_weight(e):
        doc_pre_freq, prob_M_bar_D = e
        return dict_map(lambda x:x *prob_M_bar_D, doc_pre_freq)

    weighted_doc_tf = lmap(apply_weight, zip(docs_rel_freq, p_M_bar_D_list))

    num_doc = len(weighted_doc_tf)
    p_w_m = average_tf_over_docs(weighted_doc_tf, num_doc)

    return p_w_m, num_doc


def re_tokenize(tokens):
    out = []
    for term in tokens:
        spliter = "-"
        if spliter in term and term.find(spliter) > 0:
            out.extend(term.split(spliter))
        else:
            out.append(term)
    return set(out)




def build_binary_feature(datapoint_list):
    ci = CollectionInterface()
    not_found_set = set()
    _, clue12_13_df = load_clueweb12_B13_termstat()
    cdf = 50 * 1000 * 1000

    def idf_scorer(doc, claim_text, perspective_text):
        cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        cp_tokens = set(cp_tokens)
        mentioned_terms = lfilter(lambda x: x in doc, cp_tokens)
        mentioned_terms = re_tokenize(mentioned_terms)

        def idf(term):
            if term not in clue12_13_df:
                if term in string.printable:
                    return 0
                not_found_set.add(term)

            return math.log((cdf+0.5)/(clue12_13_df[term]+0.5))

        score = sum(lmap(idf, mentioned_terms))
        max_score = sum(lmap(idf, cp_tokens))
        return score > max_score * 0.8

    def data_point_to_feature(data_point):
        label, cid, pid, claim_text, p_text = data_point
        return get_feature_binary_model(cid, pid, claim_text, p_text, ci, idf_scorer), label

    r = lmap_w_exception(data_point_to_feature, datapoint_list, KeyError)
    return r


def build_weighted_feature(datapoint_list):
    ci = CollectionInterface()

    def data_point_to_feature(data_point):
        label, cid, pid, claim_text, p_text = data_point
        return get_feature_weighted_model(cid, pid, claim_text, p_text, ci), label

    return lmap_w_exception(data_point_to_feature, datapoint_list, KeyError)


def load_all_ranked_list(disk_name):
    d = {}
    for idx in train_query_indices:
        file_name = "{}_{}.txt".format(disk_name, idx)
        file_path = os.path.join(ranked_list_save_root, file_name)
        d.update(load_galago_ranked_list(file_path))

    return d


def learn_lm(feature_label_list):
    pos_insts = lfilter(lambda x: x[1] == "1", feature_label_list)
    neg_insts = lfilter(lambda x: x[1] == "0", feature_label_list)

    def do_average(insts):
        return average_tf_over_docs(left(left(insts)), len(insts))

    pos_lm = do_average(pos_insts)
    neg_lm = do_average(neg_insts)


    classifier = LMClassifier(pos_lm, neg_lm)

    xy = lmap(lambda x: (x[0][0], int(x[1])), feature_label_list)
    classifier.tune_alpha(xy)
    return classifier


def mention_num_classifier(train, val):
    pos_insts = lfilter(lambda x: x[1] == "1", train)
    neg_insts = lfilter(lambda x: x[1] == "0", train)
    print(len(train), len(pos_insts), len(neg_insts))

    n_mention_pos = right(left(pos_insts))
    n_mention_neg = right(left(neg_insts))

    all_x = right(left(train))
    all_y = lmap(lambda x: x == "1", right(train))
    alpha = median(all_x)

    print("pos avg:", average(n_mention_pos), std(n_mention_pos))
    print("neg avg:", average(n_mention_neg), std(n_mention_neg))

    def eval(all_x, all_y):
        pred_y = lmap(lambda x: x > alpha, all_x)
        is_correct = np.equal(all_y, pred_y)
        return average(is_correct)

    print('train accuracy : ', eval(all_x, all_y))
    all_x = right(left(val))
    all_y = lmap(lambda x: x == "1", right(val))
    print('val accuracy : ', eval(all_x, all_y))
