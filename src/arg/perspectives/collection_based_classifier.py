import collections
from typing import Iterable, Counter

import math
import nltk

from arg.perspectives.clueweb_helper import preload_tf, load_tf
from arg.perspectives.pc_run_path import train_query_indices
from cache import load_from_pickle
from galagos.basic import merge_ranked_list_list
from misc_lib import lfilter, l_to_map, lmap, dict_map
from sydney_clueweb.clue_path import index_name_list
from tlm.ukp.data_gen.ukp_gen_selective import load_ranked_list


def dirichlet_smoothing(tf, dl, c_tf, c_ctf):
    mu = 1500
    denom = tf + mu * (c_tf / c_ctf)
    nom = dl + mu
    return denom / nom


class CollectionInterface:
    def __init__(self, do_lowercase=True):
        self.available_disk_list = index_name_list[:1]
        self.ranked_list = l_to_map(load_all_ranked_list, self.available_disk_list)
        self.do_lowercase = do_lowercase
        self.collection_tf = load_from_pickle("collection_tf")
        self.collection_ctf = sum(self.collection_tf.values())

    def get_ranked_documents_tf(self, claim_id, perspective_id) -> Iterable[Counter]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        ranked_list = self.get_ranked_list(query_id)
        doc_ids = lmap(lambda x:x[0], ranked_list)
        preload_tf(doc_ids)

        def do_load_tf(doc_id):
            counter = load_tf(doc_id)
            if self.do_lowercase:
                counter = {key.lower(): value for key, value in counter.items()}
            return counter

        return lmap(do_load_tf, doc_ids)

    def get_ranked_list(self, query_id):
        ranked_list_list = []
        for disk_name in self.available_disk_list:
            try:
                ranked_list = self.ranked_list[disk_name][query_id]
                ranked_list_list.append(ranked_list)
            except KeyError:
                pass

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
                             collection_interface: CollectionInterface,
                             mention_scorer,
                             threshold=0.5,
                             ):

    def is_mention(doc: Counter):
        score = mention_scorer(doc, claim_id, perspective_id)
        return score > threshold

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



# input : List[(claim_id, perspective_id)]
# output : classifier
# TODO for each claim, perspective, get_feature
# TODO
def build_feature(datapoint_list):
    ci = CollectionInterface()

    def data_point_to_feature(data_point):
        label, cid, pid, claim_text, p_text = data_point
        return get_feature_weighted_model(cid, pid, claim_text, p_text, ci)

    return lmap(data_point_to_feature, datapoint_list)


def load_all_ranked_list(disk_name):
    d = {}
    for idx in train_query_indices:
        file_name = "{}_{}".format(disk_name, idx)
        d.update(load_ranked_list(file_name))

    return d
