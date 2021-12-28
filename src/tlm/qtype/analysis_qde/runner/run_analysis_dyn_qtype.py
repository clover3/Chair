import os
import pickle
from typing import Dict

from cache import load_from_pickle, save_to_pickle
from cpath import output_path
from epath import job_man_dir
from misc_lib import tprint
from tlm.qtype.analysis_fde.analysis_a import enum_count_query
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import show_func_word_avg_embeddings, load_parse, \
    build_qtype_desc
from tlm.qtype.analysis_qde.analysis_a import qtype_analysis_a, cluster_avg_embeddings
from tlm.qtype.analysis_qde.analysis_b import analysis_b
from tlm.qtype.analysis_qde.analysis_c import run_qtype_analysis_c
from tlm.qtype.analysis_qde.variance_analysis import print_variance
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo, load_query_info_dict


def main():
    # pred_path = sys.argv[1]
    # info_path = sys.argv[2]
    split = "train"
    # qtype_entries, query_info_dict = load_parse(info_path, pred_path, split)
    # obj = qtype_entries, query_info_dict
    # save_to_pickle(obj, "run_analysis_dyn_qtype")
    tprint("Loading pickle...")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    # factor_list = dimension_normalization(qtype_entries)
    # save_to_pickle(factor_list, "factor_list")
    factor_list = load_from_pickle("factor_list")
    pos_known, neg_known = show_func_word_avg_embeddings(qtype_entries, query_info_dict, factor_list, split)
    print("{}/{} dimensions are described".format(len(pos_known), len(neg_known)))
    analysis_b(qtype_entries, query_info_dict, (pos_known, neg_known), factor_list)


def run_save_to_pickle():
    pred_path = os.path.join(output_path, "qtype", "qtype_2T_v_train")
    info_path = os.path.join(job_man_dir, )
    split = "train"
    qtype_entries, query_info_dict = load_parse(info_path, pred_path, split)
    obj = qtype_entries, query_info_dict
    save_to_pickle(obj, "run_analysis_dyn_qtype")


def run_print_variance():
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    print_variance(qtype_entries, query_info_dict)


def run_analysis_a():
    split = "train"
    tprint("Loading pickle...")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    # enum_queries(qtype_entries, query_info_dict)
    factor_list = load_from_pickle("factor_list")
    # pos_known, neg_known = show_func_word_avg_embeddings(qtype_entries, query_info_dict, factor_list, split)
    # print("{}/{} dimensions are described".format(len(pos_known), len(neg_known)))
    qtype_analysis_a(qtype_entries, query_info_dict, factor_list)


def run_analysis_c():
    tprint("Loading pickle...")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    run_qtype_analysis_c(qtype_entries, query_info_dict)


def run_clustering():
    split = "train"
    tprint("Loading pickle...")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    print("{} entries, {} info entries".format(len(qtype_entries), len(query_info_dict)))
    qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    cluster_avg_embeddings(qtype_embedding_paired)


def run_enum_queries():
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    print("{} entries".format(len(qtype_entries)))
    # enum_queries(qtype_entries, query_info_dict)
    enum_count_query(qtype_entries, query_info_dict)


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
    known_qtype_ids = show_func_word_avg_embeddings(qtype_entries, query_info_dict, split)
    # run_qtype_analysis(qtype_entries, query_info_dict, known_qtype_ids)


if __name__ == "__main__":
    run_enum_queries()
