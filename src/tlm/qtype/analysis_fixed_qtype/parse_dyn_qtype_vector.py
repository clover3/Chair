import random
from collections import Counter, defaultdict
from typing import Iterator
from typing import List, Dict, Tuple

import numpy as np

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from misc_lib import group_by, get_second, tprint, TimeEstimator
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_qde.analysis_common import get_avg_vector
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import load_query_info_dict, QueryInfo, \
    structured_qtype_text
from tlm.qtype.qtype_instance import QTypeInstance


def parse_q_weight_output(raw_prediction_path, data_info) -> List[QTypeInstance]:
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    for e in viewer:
        yield parse_q_weight_inner(data_info, e)


def parse_q_weight_inner(data_info, e):
    data_id = e.get_vector("data_id")[0]
    label_ids = e.get_vector("label_ids")[0]
    de_input_ids = e.get_vector("de_input_ids")
    logits = e.get_vector("logits")
    qtype_vector_qe = e.get_vector("qtype_vector_qe")
    qtype_vector_de = e.get_vector("qtype_vector_de")
    info_entry = data_info[str(data_id)]
    query_id = info_entry['query'].query_id
    doc_id = info_entry['candidate'].id
    try:
        passage_idx = info_entry['passage_idx']
    except KeyError:
        passage_idx = -1
    inst = QTypeInstance(query_id, doc_id, passage_idx,
                         de_input_ids, qtype_vector_qe, qtype_vector_de,
                         label_ids, logits,
                         e.get_vector("bias"),
                         e.get_vector("q_bias"),
                         e.get_vector("d_bias"),
                         )
    return inst


def build_qtype_desc(qtype_entries: Iterator[QTypeInstance], query_info_dict: Dict[str, QueryInfo])\
        -> Tuple[List[Tuple[str, np.array]], Dict[str, int]]:
    def get_func_str(e: QTypeInstance) -> str:
        func_str = " ".join(query_info_dict[e.qid].functional_tokens)
        return func_str

    grouped = group_by(qtype_entries, get_func_str)
    qtype_embedding = []
    n_query = {}
    for func_str, items in grouped.items():
        avg_vector = np.mean(np.stack([e.qtype_weights_qe for e in items], axis=0), axis=0)
        qtype_embedding.append((func_str, avg_vector))
        n_query[func_str] = len(items)

    return qtype_embedding, n_query


def merge_vectors_by_frequency(embedding_list):
    def get_vector_diff(v1, v2):
        mag = (sum(np.abs(v1)) + sum(np.abs(v2))) / 2
        error = np.sum(np.abs(v1 - v2))
        rel_error = error / mag
        return error, rel_error


    def sig(embedding):
        return np.sum(np.abs(embedding))
    embedding_list.sort(key=sig)
    prev_v = None
    v_idx = 0
    v_list = []
    cnt = Counter()
    for e in embedding_list:
        if prev_v is not None:
            error, rel_error = get_vector_diff(prev_v, e)
            if rel_error < 1e-2:
                cnt[v_idx] += 1
            else:
                prev_v = e
                v_idx += 1
                v_list.append(e)
                cnt[v_idx] += 1
        else:
            prev_v = e
            v_list.append(prev_v)
            cnt[v_idx] += 1

    for key, _ in cnt.most_common(1):
        return v_list[key]

    raise Exception()


def build_qtype_embedding_old(qtype_entries: Iterator[QTypeInstance], query_info_dict: Dict[str, QueryInfo]) \
        -> Dict[str, np.array]:
    func_str_to_vector = {}
    seen = set()
    for e in qtype_entries:
        func_str = " ".join(query_info_dict[e.qid].functional_tokens)
        if func_str in seen:
            continue

        seen.add(func_str)
        func_str_to_vector[func_str] = e.qtype_weights_qe
    return func_str_to_vector


def show_vector_distribution(v):
    step = 0.1
    st = -1
    while st < 1:
        f = np.logical_and(np.less_equal(st, v), np.less(v, st+step))
        n = np.count_nonzero(f)
        if n:
            print("[{0:.1f},{1:.1f}]: {2}".format(st, st+step, n))
        st += step


def avg_vector_from_qtype_entries(qtype_entries):
    vector_list = []
    for e in qtype_entries:
        vector_list.append(e.qtype_weights_qe)

    return get_avg_vector(vector_list, vector_list[0])


def show_func_word_avg_embeddings(qtype_entries, query_info_dict,
                                  factor_list,
                                  split):
    print("Building qtype desc")
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    random.shuffle(qtype_entries)

    n_sample = 10000 * 10
    print("Sample {} from {}".format(n_sample, len(qtype_entries)))
    qtype_entries = qtype_entries[:n_sample]
    g_avg_vector = avg_vector_from_qtype_entries(qtype_entries)

    qtype_embedding_paired: List[Tuple[str, np.array]] = qtype_embedding_paired
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    mapping = structured_qtype_text(query_info_dict)
    qtype_embedding_paired_new = []
    for func_str, avg_vector in qtype_embedding_paired:
        head, tail = mapping[func_str]
        if head and tail:
            s = "{0} [] {1}".format(head, tail)
        else:
            s = head + tail
        qtype_embedding_paired_new.append((s, avg_vector))

    qtype_embedding_paired = qtype_embedding_paired_new

    dim_printed_counter_pos, grouped_by_dim = group_func_by_dim_value(qtype_embedding_paired, factor_list)
    dim_printed_counter_neg, grouped_by_dim_neg = group_func_by_dim_value_neg(qtype_embedding_paired, factor_list)

    for dim_idx, cnt in dim_printed_counter_pos.most_common(100):
        print("Dim {} appeared {}".format(dim_idx, cnt))

    pos_known = list(grouped_by_dim.keys())
    neg_known = list(grouped_by_dim_neg.keys())

    keys = list(grouped_by_dim.keys())
    keys.sort(key=lambda k: len(grouped_by_dim[k]))
    for k in keys:
        pos_items = grouped_by_dim[k]
        neg_items = grouped_by_dim_neg[k]

        print("Dim {} : #func_str: {}/{} ".format(k, len(pos_items), len(neg_items)))
        pos_items.sort(key=get_second, reverse=True)
        neg_items.sort(key=get_second)

        some_pos_items = pos_items[:30]
        some_neg_items = neg_items[:30]
        def print_items(items):
            print(" / ".join(["{0} ({1:.2f})".format(func_str, v) for func_str, v in items]))

        print_items(some_pos_items)
        print_items(some_neg_items)

    known_qtype_ids = keys
    return pos_known, neg_known


def group_func_by_dim_value(qtype_embedding_paired, factor_list):
    dim_printed_counter = Counter()
    grouped_by_dim = defaultdict(list)
    factor_np = np.array(factor_list)
    for func_str, avg_vector in qtype_embedding_paired:
        vector = avg_vector / factor_np
        rank = np.argsort(vector)[::-1]
        rep = ""
        for d_idx in rank[:10]:
            v = vector[d_idx]
            if v > 0.5:
                dim_printed_counter[d_idx] += 1
                rep += "{0}: {1:.2f} /".format(d_idx, v)
                grouped_by_dim[d_idx].append((func_str, v))
        # print(func_str, rep)
    return dim_printed_counter, grouped_by_dim


def group_func_by_dim_value_neg(qtype_embedding_paired, factor_list):
    dim_printed_counter = Counter()
    grouped_by_dim = defaultdict(list)
    for func_str, avg_vector in qtype_embedding_paired:
        rank = np.argsort(avg_vector)
        rep = ""
        for d_idx in rank[:5]:
            v = avg_vector[d_idx]
            v = v / factor_list[d_idx]
            if v < -0.5:
                dim_printed_counter[d_idx] += 1
                rep += "{0}: {1:.2f} /".format(d_idx, v)
                grouped_by_dim[d_idx].append((func_str, v))
    return dim_printed_counter, grouped_by_dim


def dimension_normalization(qtype_entries):
    tprint("dimension_normalization")
    n_dim = len(qtype_entries[0].qtype_weights_qe)
    factor_list = []
    ticker = TimeEstimator(n_dim)
    for dim_id in range(n_dim):
        v_list = []
        ticker.tick()
        for e in qtype_entries:
            v = e.qtype_weights_qe[dim_id]
            v_list.append(v)
        v_list.sort()

        n_step = 10
        head_step = 1
        tail_step = n_step - 1
        head_idx = int((head_step / n_step) * len(v_list))
        head_v = v_list[head_idx]
        tail_idx = int((tail_step / n_step) * len(v_list))
        tail_v = v_list[tail_idx]
        if abs(head_v) > abs(tail_v):
            factor = head_v
        else:
            factor = tail_v
        factor_list.append(factor)
        out_s = ""
        for step in range(n_step):
            idx = int((step / n_step) * len(v_list))
            v = v_list[idx] / (factor + 1e-8)
            out_s += "{0}: {1:.2f} ".format(step, v)

        v = v_list[-1] / (factor + 1e-8)
        out_s += "{0}: {1:.2f} ".format(-1, v)

        # print("{0} {1} ".format(dim_id, out_s))
    return factor_list


def load_parse(info_file_path, raw_prediction_path, split):
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Parsing predictions...")
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    print("Reading QType Entries")
    qtype_entries: List[QTypeInstance] = list(parse_q_weight_output(raw_prediction_path, info))
    return qtype_entries, query_info_dict
