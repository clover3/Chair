import random
from collections import Counter, defaultdict
from typing import Iterator
from typing import List, Dict, Tuple

import numpy as np

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from list_lib import left, right
from misc_lib import group_by, get_second, tprint, TimeEstimator
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_qde.analysis_common import get_avg_vector
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list, convert_ids_to_tokens
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import load_query_info_dict, QueryInfo, \
    structured_qtype_text
from tlm.qtype.qtype_instance import QTypeInstance


def parse_q_weight_output(raw_prediction_path, data_info) -> List[QTypeInstance]:
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    for e in viewer:
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
        yield inst


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


def show_qtype_embeddings(qtype_entries, query_info_dict, split):
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
    dim_printed_counter = Counter()
    grouped_by_dim = defaultdict(list)
    scaling = 100
    for func_str, avg_vector in qtype_embedding_paired:
        rank = np.argsort(avg_vector)[::-1]
        rep = ""
        cut = np.max(avg_vector) * 0.3
        for d_idx in rank[:5]:
            v = avg_vector[d_idx]
            if abs(v) > 1:
                dim_printed_counter[d_idx] += 1
                rep += "{0}: {1:.2f} /".format(d_idx, v)
                grouped_by_dim[d_idx].append((func_str, v))
        # print(func_str, rep)

    for dim_idx, cnt in dim_printed_counter.most_common(100):
        print("Dim {} appeared {}".format(dim_idx, cnt))

    keys = list(grouped_by_dim.keys())
    keys.sort(key=lambda k: len(grouped_by_dim[k]))
    for k in keys:
        items = grouped_by_dim[k]
        items.sort(key=get_second, reverse=True)
        n_items = len(items)
        items = items[:30]
        print("Dim {} : {} func_str".format(k, n_items))
        print(" / ".join(["{0} ({1:.2f})".format(func_str, v * 100) for func_str, v in items]))
    known_qtype_ids = keys
    return known_qtype_ids



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
    return factor


def run_qtype_analysis(qtype_entries: List[QTypeInstance], query_info_dict, known_qtype_ids):
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    print("Building qtype desc")
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    print("Done")
    n_func_word = len(qtype_embedding_paired)
    func_type_id_to_text: List[str] = left(qtype_embedding_paired)
    qtype_embedding_np = np.stack(right(qtype_embedding_paired), axis=0)
    threshold = 0.05
    def print_by_dimension(v):
        rank = np.argsort(v)[::-1]
        for i in rank:
            value = v[i]
            if value < threshold:
                break
            else:
                if i in known_qtype_ids:
                    print("{0}: {1:.2f}".format(i, value))

    class RowPrinter:
        def __init__(self, func_word_weights):
            self.r_unit = 0.1
            self.cur_range = 10 * self.r_unit
            self.text_for_cur_line = []
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

            self.text_for_cur_line.append("[{}]{}".format(idx, func_text))

        def pop(self):
            print("{0:.1f}~ : ".format(self.cur_range) + " / ".join(self.text_for_cur_line))
            self.text_for_cur_line = []

    print_per_qid = Counter()
    n_print = 0
    for e_idx, e in enumerate(qtype_entries):
        display = False
        if e_idx % 10 and print_per_qid[e.qid] < 4:
            # display = True
            pass
        #  : Show qid/query text
        #  : Show non-zero score of document
        if e.logits > 3 or e.d_bias > 3:
            why_display = "Display by high logits"
            display = True
        if e.label:
            why_display = "Display by true label"
            display = True
        if not display:
            continue
        print_per_qid[e.qid] += 1

        q_rep = " ".join(query_info_dict[e.qid].out_s_list)

        def cossim(nd2, nd1):
            a =  np.matmul(nd2, nd1)
            b = (np.linalg.norm(nd2, axis=1) * np.linalg.norm(nd1))
            return a / b
        func_word_weights_q = cossim(qtype_embedding_np, e.qtype_weights_qe)
        func_word_weights_d = cossim(qtype_embedding_np, e.qtype_weights_de)

        scale = np.max(e.qtype_weights_qe)
        scale = 1/100
        scaled_query_qtype = e.qtype_weights_qe / scale
        scaled_document_qtype = e.qtype_weights_de * scale
        display = False
        for key in known_qtype_ids:
            if scaled_document_qtype[key] > threshold:
                display = True
        if not display:
            continue
        n_print += 1
        if n_print % 5 == 0:
            dummy = input("Enter something ")

        print("---------------------------------")
        print()
        print(e.qid, q_rep)
        print("{} - {}".format(e.doc_id, "Relevant" if e.label else "Non-relevant"))
        print("Score bias q_bias d_bias")
        print(" ".join(map("{0:.2f}".format, [e.logits, e.bias, e.q_bias, e.d_bias])))
        print("QType Query ")
        row_printer = RowPrinter(func_word_weights_q)
        for i in range(20):
            row_printer.do(i)
        row_printer.pop()
        print("Raw QWeights")

        show_vector_distribution(scaled_query_qtype)
        print_by_dimension(scaled_query_qtype)
        print("Top QType from Doc")
        print(n_func_word)
        row_printer = RowPrinter(func_word_weights_d)
        for i in range(20):
            row_printer.do(i)
        row_printer.pop()
        print("< Skip middle >")
        for i in range(n_func_word - 20, n_func_word):
            row_printer.do(i)
        row_printer.pop()

        print("Raw DWeights")
        show_vector_distribution(scaled_document_qtype)
        print_by_dimension(scaled_document_qtype)

        print(e.qtype_weights_de)
        seg1, seg2 = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)
        seg2_tokens = convert_ids_to_tokens(voca_list, seg2)

        passage: str = pretty_tokens(seg2_tokens, True)
        print(passage)


def load_parse(info_file_path, raw_prediction_path, split):
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Parsing predictions...")
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    print("Reading QType Entries")
    qtype_entries: List[QTypeInstance] = list(parse_q_weight_output(raw_prediction_path, info))
    return qtype_entries, query_info_dict
