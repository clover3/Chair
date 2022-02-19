from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import sklearn.cluster
import spacy

from cache import load_from_pickle
from list_lib import left, right
from misc_lib import group_by, tprint
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import build_qtype_desc, show_vector_distribution
from tlm.qtype.analysis_qde.analysis_common import get_passsage_fn
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import structured_qtype_text
from tlm.qtype.qtype_instance import QTypeInstance


def enum_queries(qtype_entries: List[QTypeInstance], query_info_dict):
    seen = set()
    for e_idx, e in enumerate(qtype_entries):
        q_rep = " ".join(query_info_dict[e.qid].out_s_list)

        if q_rep not in seen:
            print(q_rep)
            seen.add(q_rep)


class RangeGrouping:
    def __init__(self):
        self.r_unit = 0.1
        self.cur_range = 100 * self.r_unit
        self.cur_group_idx = []
        self.groups = []

    def do(self, idx, w):
        while self.cur_range - w > self.r_unit:
            if self.cur_group_idx:
                self.pop()
            self.cur_range = self.cur_range - self.r_unit
        self.cur_group_idx.append(idx)

    def pop(self):
        self.groups.append((self.cur_range, self.cur_group_idx))
        self.cur_group_idx = []


def qtype_analysis_a(qtype_entries: List[QTypeInstance],
                     query_info_dict,
                     scale_factor_list):
    scale_factor = np.array(scale_factor_list)
    get_passage = get_passsage_fn()
    print("Building qtype desc")
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    func_word_structured_d: Dict[str, Tuple[str, str]] = structured_qtype_text(query_info_dict)
    # print("Clustering...")
    # clusters = cluster_avg_embeddings(qtype_embedding_paired)
    # save_to_pickle(clusters, "run_analysis_dyn_qtype_cluster")
    clusters = load_from_pickle("run_analysis_dyn_qtype_cluster")
    n_func_word = len(qtype_embedding_paired)
    func_type_id_to_text: List[str] = left(qtype_embedding_paired)
    qtype_embedding_np = np.stack(right(qtype_embedding_paired), axis=0)
    nlp = spacy.load("en_core_web_sm")

    is_person_query_d = {}
    def is_person_query(info):
        query = info.query
        if query not in is_person_query_d:
            doc = nlp(query)
            appropriate = False
            for e in doc.ents:
                if e.text == info.content_span and e.label_ == "PERSON":
                    appropriate = True
            is_person_query_d[query] = appropriate

        return is_person_query_d[query]


    class RowPrinter:
        def __init__(self, func_word_weights):
            self.r_unit = 0.1
            self.cur_range = 100 * self.r_unit
            self.text_for_cur_line = []
            self.cur_group_idx = []
            self.func_word_weights = func_word_weights
            self.ranked = np.argsort(func_word_weights)[::-1]

        def do(self, i):
            idx = self.ranked[i]
            func_text = func_type_id_to_text[idx]
            w = self.func_word_weights[idx]

            while self.cur_range - w > self.r_unit:
                if self.text_for_cur_line:
                    self.pop()
                self.cur_range = self.cur_range - self.r_unit
            self.cur_group_idx.append(idx)

            self.text_for_cur_line.append("[{}]{}".format(idx, func_text))

        def pop(self):
            print("{0:.1f}~ : ".format(self.cur_range) + " / ".join(self.text_for_cur_line))
            self.text_for_cur_line = []

    target_content_words = ["sodium",
                            "leak detection",
                            "sociology college of science",
                            "mclean",
                            "mdu stock",
                            "james lange theory",
                            "isp",
                            "trevenant",
                            "ithaca",
                            "lupus nephritis",
                            "parabolic"]

    print_per_qid = Counter()
    n_print = 0
    for e_idx, e in enumerate(qtype_entries):
        if e_idx % 10 and print_per_qid[e.qid] < 4:
            pass

        f_high_logit = e.logits > 0 or e.d_bias > 0
        f_relevant = e.label
        info = query_info_dict[e.qid]

        display = is_person_query(info)
        f_target_keyword = info.content_span in target_content_words
        content_tokens = info.content_span.split()
        f_short_content = len(content_tokens) < 3
        print_per_qid[e.qid] += 1
        q_rep = " ".join(info.out_s_list)
        func_word_weights_q = np.matmul(qtype_embedding_np, e.qtype_weights_qe)
        func_word_weights_d = np.matmul(qtype_embedding_np, e.qtype_weights_de)

        scaled_document_qtype = e.qtype_weights_de * scale_factor
        max_idx = np.argmax(func_word_weights_d)
        # if max(func_word_weights_d) < 2:
        #     display = False
        if not display:
            continue

        n_print += 1
        if n_print % 5 == 0:
            dummy = input("Enter something ")

        print("---------------------------------")
        print()
        print(e.qid, q_rep)
        print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))
        print("Score bias q_bias d_bias")
        print(" ".join(map("{0:.2f}".format, [e.logits, e.bias, e.q_bias, e.d_bias])))
        # print("Similar func_word - Query ")
        # row_printer = RowPrinter(func_word_weights_q)
        # for i in range(20):
        #     row_printer.do(i)
        # row_printer.pop()
        print("Max {} at {}".format(func_word_weights_d[max_idx], max_idx))
        print("QWeights")
        print("Similar func_word - Doc")
        print(n_func_word)
        grouping = RangeGrouping()
        rank = np.argsort(func_word_weights_d)[::-1]
        for i in rank:
            grouping.do(i, func_word_weights_d[i])
        grouping.pop()

        seen_cluster = set()
        for range_, group in grouping.groups:
            if range_ < 0:
                break

            group_item_text = []
            for idx in group:
                cluster_idx = clusters[idx]
                if cluster_idx in seen_cluster:
                    pass
                else:
                    seen_cluster.add(cluster_idx)
                    func_text = func_word_structured_d[func_type_id_to_text[idx]]
                    group_item_text.append("[{}]{} ({})".format(idx, func_text, cluster_idx))

            if group_item_text:
                print("{0:.1f}~ : ".format(range_) + " / ".join(group_item_text))

        print("Raw DWeights")
        show_vector_distribution(scaled_document_qtype)
        print(e.qtype_weights_de)
        print(get_passage(e.de_input_ids))


def cluster_avg_embeddings(qtype_embedding_paired):
    # cluster_model = sklearn.cluster.KMeans(n_clusters=100, tol=1e-7)
    cluster_model = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=2)
    n_func_word = len(qtype_embedding_paired)
    tprint("Start clustering with {} functional spans".format(n_func_word))
    func_type_id_to_text: List[str] = left(qtype_embedding_paired)
    qtype_embedding_np = np.stack(right(qtype_embedding_paired), axis=0)
    res = cluster_model.fit(qtype_embedding_np)
    maybe_n_cluster = max(res.labels_)+1
    tprint("{} cluster found".format(maybe_n_cluster))

    def get_group(idx):
        return res.labels_[idx]

    grouped = group_by(list(range(n_func_word)), get_group)
    keys = list(grouped.keys())
    keys.sort(key=lambda k: len(grouped[k]), reverse=True)
    for key in keys:
        items = grouped[key]
        print("Cluster {} : {} items".format(key, len(items)))
        func_texts_for_cluster = [func_type_id_to_text[j] for j in items]
        func_texts_for_cluster.sort(key=len)
        rep = " / ".join(func_texts_for_cluster)
        print(rep)

    return res.labels_
