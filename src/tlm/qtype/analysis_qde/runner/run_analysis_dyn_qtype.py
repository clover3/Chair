import os
import pickle
from typing import Dict

from cache import load_from_pickle
from cpath import output_path
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import show_qtype_embeddings, load_parse
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict


def main():
    # pred_path = sys.argv[1]
    # info_path = sys.argv[2]
    split = "train"
    # qtype_entries, query_info_dict = load_parse(info_path, pred_path, split)
    # obj = qtype_entries, query_info_dict
    # save_to_pickle(obj, "run_analysis_dyn_qtype")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    known_qtype_ids = show_qtype_embeddings(qtype_entries, query_info_dict, split)
    # run_qtype_analysis(qtype_entries, query_info_dict, known_qtype_ids)


def qtype_2U_train():
    split = "train"
    pred_path = os.path.join(output_path, "qtype_2U_v_train")
    info_path = os.path.join(output_path, "MMD_train_qe_de_distill_10doc_info_conv")
    qtype_entries, query_info_dict = load_parse(info_path, pred_path, split)
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    middle = int(len(qtype_entries) / 2)
    obj_head = qtype_entries[:middle]
    obj_tail = qtype_entries[middle:]
    def dump(obj, name):
        root_dir = "D:\\data\\chair_output"
        pickle.dump(obj, open(os.path.join(root_dir, name), "wb"))
    dump(obj_head, "run_analysis_qtype_2U_v_train_head")
    dump(obj_tail, "run_analysis_qtype_2U_v_train_tail")


def qtype_2U_train_cached():
    def load(name):
        root_dir = "D:\\data\\chair_output"
        return pickle.load(open(os.path.join(root_dir, name), "rb"))
    split = "train"
    qtype_entries = []
    qtype_entries.extend(load("run_analysis_qtype_2U_v_train_head"))
    qtype_entries.extend(load("run_analysis_qtype_2U_v_train_tail"))
    query_info_dict: Dict[str, QueryInfo] = load_query_info_dict(split)
    # qtype_entries, query_info_dict = load_from_pickle("run_analysis_qtype_2U_v_train")
    known_qtype_ids = show_qtype_embeddings(qtype_entries, query_info_dict, split)
    # run_qtype_analysis(qtype_entries, query_info_dict, known_qtype_ids)


if __name__ == "__main__":
    qtype_2U_train_cached()
