from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from cache import load_from_pickle
from list_lib import left, right
from misc_lib import find_max_idx
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import build_qtype_desc
from tlm.qtype.analysis_fixed_qtype.parse_qtype_vector import QDistClient
from tlm.qtype.analysis_qde.analysis_common import get_passsage_fn
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import structured_qtype_text, QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


def run_qtype_analysis_c(qtype_entries: List[QTypeInstance],
                         query_info_dict: Dict[str, QueryInfo],
                       ):
    qdist = QDistClient()
    get_passage = get_passsage_fn()
    print("Building qtype desc")
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    func_word_structured_d: Dict[str, Tuple[str, str]] = structured_qtype_text(query_info_dict)

    # print("Clustering...")
    # clusters = cluster_avg_embeddings(qtype_embedding_paired)
    # save_to_pickle(clusters, "run_analysis_dyn_qtype_cluster")
    clusters: Dict[int, int] = load_from_pickle("run_analysis_dyn_qtype_cluster")
    cluster_member_list = defaultdict(list)
    for type_idx, cluster_idx in enumerate(clusters):
        cluster_member_list[cluster_idx].append(type_idx)


    n_func_word = len(qtype_embedding_paired)
    func_type_id_to_text: List[str] = left(qtype_embedding_paired)
    qtype_embedding_np = np.stack(right(qtype_embedding_paired), axis=0)
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    qtype_pred_n_class = 2048
    qtype_id_conv_map = {}
    for i in range(n_func_word):
        func_word = func_type_id_to_text[i]
        try:
            qtype_id = qtype_id_mapping[func_word]
            if qtype_id < qtype_pred_n_class:
                qtype_id_conv_map[i] = qtype_id
        except KeyError:
            pass

    def conv_qtype_distrib(v):
        default_prob = np.log(1e-8)
        new_v = []
        for i in range(n_func_word):
            try:
                new_v.append(v[qtype_id_conv_map[i]])
            except KeyError:
                new_v.append(default_prob)
        return np.stack(new_v)

    def get_log_func_word_likely(content_span):
        func_word_likely = qdist.query(content_span)
        log_func_word_likely = np.log(func_word_likely)
        log_func_word_likely = conv_qtype_distrib(log_func_word_likely)
        return log_func_word_likely

    class RowPrinter:
        def __init__(self, vector):
            self.r_unit = 0.1
            self.cur_range = 100 * self.r_unit
            self.text_for_cur_line = []
            self.cur_group_idx = []
            self.func_word_weights = vector
            self.ranked = np.argsort(vector)[::-1]

        def do(self, i):
            idx = self.ranked[i]
            func_text = func_type_id_to_text[idx]
            w = self.func_word_weights[idx]

            while self.cur_range - w > self.r_unit:
                if self.text_for_cur_line:
                    self.pop()
                self.cur_range = self.cur_range - self.r_unit
            self.cur_group_idx.append(idx)

            self.text_for_cur_line.append("[{}]{}".format(idx, func_word_structured_d[func_text]))

        def pop(self):
            print("{0:.1f}~ : ".format(self.cur_range) + " / ".join(self.text_for_cur_line))
            self.text_for_cur_line = []

    threshold_1 = 3
    threshold_2 = 0
    func_likely_base = np.log(1 / qtype_pred_n_class)
    for e_idx, e in enumerate(qtype_entries):
        f_high_logit = e.logits > 0 or e.d_bias > 3
        if not f_high_logit:
            continue

        cur_q_info = query_info_dict[e.qid]
        q_rep = " ".join(cur_q_info.out_s_list)
        func_word_weights_d = np.matmul(qtype_embedding_np, e.qtype_weights_de)

        n_promising_func_word = np.count_nonzero(np.less(threshold_1, func_word_weights_d))
        if not n_promising_func_word:
            continue

        log_func_word_likely = get_log_func_word_likely(cur_q_info.content_span)
        combined_score = func_word_weights_d + log_func_word_likely - func_likely_base

        n_promising_func_word2 = np.count_nonzero(np.less(threshold_2, combined_score))
        if not n_promising_func_word2:
            continue
        print("---------------------------------")
        print("e_idx=", e_idx)
        print(e.qid, q_rep)
        print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))
        print("Score bias q_bias d_bias")
        print(" ".join(map("{0:.2f}".format, [e.logits, e.bias, e.q_bias, e.d_bias])))
        rank = np.argsort(combined_score)[::-1]
        seen_cluster = set()
        for i in range(100):
            type_i = rank[i]
            cluster_idx = clusters[type_i]
            if cluster_idx in seen_cluster:
                continue

            seen_cluster.add(cluster_idx)
            func_text = func_type_id_to_text[type_i]
            s = "{0} {1:.2f} = {2:.2f} + {3:.2f}".format(
                func_word_structured_d[func_text],
                combined_score[type_i],
                func_word_weights_d[type_i],
                log_func_word_likely[type_i])
            print(s)
            best_type_i = find_max_idx(lambda type_j: log_func_word_likely[type_j], cluster_member_list[cluster_idx])
            func_text = func_type_id_to_text[best_type_i]
            s = "{0} {1:.2f} = {2:.2f} + {3:.2f}".format(
                func_word_structured_d[func_text],
                combined_score[best_type_i],
                func_word_weights_d[best_type_i],
                log_func_word_likely[best_type_i])
            print(s)

            if len(seen_cluster) > 20:
                break
        print(get_passage(e.de_input_ids))

