from collections import Counter
from typing import List, Dict
from typing import NamedTuple, Iterator

import numpy as np

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from cache import load_from_pickle
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import pretty_tokens
from dataset_specific.msmarco.common import load_queries, QueryID
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list, convert_ids_to_tokens


class QTypeInstance(NamedTuple):
    qid: str
    doc_id: str
    passage_idx: str
    content_words: List[str]
    doc: List[str]
    qtype_weights_qe: np.array
    qtype_weights_de: np.array
    label: int
    logits: float


def parse_q_weight_output(raw_prediction_path, data_info) -> List[QTypeInstance]:
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    tokenizer = viewer.tokenizer
    voca_list = get_voca_list(tokenizer)
    for e in viewer:
        data_id = e.get_vector("data_id")[0]
        label_ids = e.get_vector("label_ids")[0]
        de_input_ids = e.get_vector("de_input_ids")
        logits = e.get_vector("logits")
        qtype_vector_qe = e.get_vector("qtype_vector_qe")
        qtype_vector_de = e.get_vector("qtype_vector_de")
        def split_conv_input_ids(input_ids):
            seg1, seg2 = split_p_h_with_input_ids(input_ids, input_ids)
            seg1_tokens = convert_ids_to_tokens(voca_list, seg1)
            seg2_tokens = convert_ids_to_tokens(voca_list, seg2)
            return seg1_tokens, seg2_tokens

        info_entry = data_info[str(data_id)]
        query_id = info_entry['query'].query_id
        doc_id = info_entry['candidate'].id
        passage_idx = info_entry['passage_idx']
        entity, doc = split_conv_input_ids(de_input_ids)
        inst = QTypeInstance(query_id, doc_id, passage_idx,
                             entity, doc, qtype_vector_qe, qtype_vector_de,
                             label_ids, logits)
        yield inst


def run_qtype_analysis(raw_prediction_path, info_file_path, split):
    f_handler = get_format_handler("qc")
    print("Reading info...")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Parsing predictions...")
    qtype_id_mapping: Dict[str, int] = load_from_pickle("qtype_id_mapping")
    qtype_id_to_text: Dict[int, str] = {v: k for k, v in qtype_id_mapping.items()}
    qtype_id_to_text[0] = "Out-of-QVoca"

    qtype_entries: Iterator[QTypeInstance] = parse_q_weight_output(raw_prediction_path, info)
    query_dict: Dict[QueryID, str] = dict(load_queries(split))
    class RowPrinter:
        def __init__(self, qtype_weights):
            self.cur_range = 10
            self.text_for_cur_line = []
            self.qtype_weights = qtype_weights
            self.ranked = np.argsort(qtype_weights)[::-1]


        def do(self, i):
            idx = self.ranked[i]
            q_type_text = qtype_id_to_text[idx]
            w = self.qtype_weights[idx]

            while self.cur_range - w > 1:
                if self.text_for_cur_line:
                    self.pop()
                self.cur_range = self.cur_range - 1
            self.text_for_cur_line.append("[{}]{}".format(idx, q_type_text))

        def pop(self):
            print("{}~ : ".format(self.cur_range) + " / ".join(self.text_for_cur_line))
            self.text_for_cur_line = []

    print_per_qid = Counter()
    n_print = 0
    for e_idx, e in enumerate(qtype_entries):
        display = False
        if e_idx % 10 and print_per_qid[e.qid] < 4:
            display = True
        #  : Show qid/query text
        #  : Show non-zero score of document
        qtype_id = int(np.argmax(e.qtype_weights_qe))
        query = query_dict[QueryID(e.qid)]
        match_score = e.qtype_weights_de[qtype_id]

        if match_score > 1:
            display = True
        if e.label:
            display = True

        if not display:
            continue

        print_per_qid[e.qid] += 1
        func_words = qtype_id_to_text[qtype_id]
        func_length = len(func_words)
        content_len = len(query) - func_length
        if content_len > 10:
            continue
        n_print += 1
        if n_print % 10 == 0:
            dummy = input("Enter something ")

        print()

        q_rep = query.replace(func_words, "[{}]".format(func_words))
        print(e.qid, q_rep)
        print("{} - {}".format(e.doc_id, "Relevant" if e.label else "Non-relevant"))
        print("Score {0:.2f}".format(e.logits))
        print("QType Query ")
        row_printer = RowPrinter(e.qtype_weights_qe)
        for i in range(20):
            row_printer.do(i)
        row_printer.pop()

        print("QType_doc[{0}] = {1:.2f}".format(qtype_id, e.qtype_weights_de[qtype_id]))
        n_qtype = len(e.qtype_weights_de)
        print("Top QType from Doc")

        print(n_qtype)
        row_printer = RowPrinter(e.qtype_weights_de)
        for i in range(20):
            row_printer.do(i)
        row_printer.pop()
        print("< Skip middle >")
        for i in range(n_qtype - 20, n_qtype):
            row_printer.do(i)
        row_printer.pop()

        passage: str = pretty_tokens(e.doc, True)
        print(passage)