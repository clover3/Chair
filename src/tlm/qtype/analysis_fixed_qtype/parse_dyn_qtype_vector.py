from collections import Counter
from typing import List, Dict, Tuple
from typing import NamedTuple, Iterator

import numpy as np

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from list_lib import left, right
from misc_lib import group_by
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list, convert_ids_to_tokens
from tlm.qtype.qid_to_content_tokens import load_query_info_dict, QueryInfo


class QTypeInstance(NamedTuple):
    qid: str
    doc_id: str
    passage_idx: str
    de_input_ids: np.array
    qtype_weights_qe: np.array
    qtype_weights_de: np.array
    label: int
    logits: float
    bias: float
    q_bias: float
    d_bias: float


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
        passage_idx = info_entry['passage_idx']
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



def run_qtype_analysis(raw_prediction_path, info_file_path, split):
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Parsing predictions...")
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    print("Reading QType Entries")
    qtype_entries: List[QTypeInstance] = list(parse_q_weight_output(raw_prediction_path, info))
    print("Building qtype desc")
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    print("Done")
    n_func_word = len(qtype_embedding_paired)
    func_type_id_to_text = left(qtype_embedding_paired)
    qtype_embedding_np = np.stack(right(qtype_embedding_paired), axis=0)
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
            display = True
        #  : Show qid/query text
        #  : Show non-zero score of document
        if e.logits > 1:
            display = True
        if e.label:
            display = True

        if not display:
            continue

        print_per_qid[e.qid] += 1
        n_print += 1
        if n_print % 10 == 0:
            dummy = input("Enter something ")
        print()
        q_rep = " ".join(query_info_dict[e.qid].out_s_list)

        def cossim(nd2, nd1):
            a =  np.matmul(nd2, nd1)
            b = (np.linalg.norm(nd2, axis=1) * np.linalg.norm(nd1))
            return a / b

        func_word_weights_q = cossim(qtype_embedding_np, e.qtype_weights_qe)
        func_word_weights_d = cossim(qtype_embedding_np, e.qtype_weights_de)

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

        scale = np.max(e.qtype_weights_qe)
        show_vector_distribution(e.qtype_weights_qe / scale)

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
        show_vector_distribution(e.qtype_weights_de * scale)
        print(e.qtype_weights_de)
        seg1, seg2 = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)
        seg2_tokens = convert_ids_to_tokens(voca_list, seg2)

        passage: str = pretty_tokens(seg2_tokens, True)
        print(passage)
