import os
from typing import Dict

import numpy as np

from cache import load_from_pickle, load_pickle_from
from cpath import output_path
from misc_lib import tprint
from tlm.qtype.analysis_fde.analysis_a import run_qtype_analysis_a, enum_count_query
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype, load_q_bias


def run_analysis_a(run_name):
    tprint("Loading pickle...")
    job_id = 0
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype(run_name)
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    save_path = os.path.join(output_path, "qtype", run_name + '_sample', str(job_id))
    qtype_entries, query_info_dict = load_pickle_from(save_path)
    cluster = load_from_pickle("{}_clusters".format(run_name))
    run_qtype_analysis_a(qtype_entries, query_info_dict, q_embedding_d, q_bias_d, cluster, True)


def run_show_enum_queries():
    # qtype_entries, query_info_dict = load_from_pickle("analysis_f_de_qtype_2V_v_dev_200000")
    qtype_entries, query_info_dict = load_from_pickle("analysis_f_de_qtype_2V_v_train_200000")
    print("{} entries".format(len(qtype_entries)))
    enum_count_query(qtype_entries, query_info_dict)


def show_qtypes():
    run_name = "qtype_2X_v_train_200000"
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype(run_name)
    for key in q_embedding_d:
        print(key)


def main():
    # run_save_to_pickle_train()
    run_name = "qtype_2X_v_train_200000"
    run_name = "qtype_2Y_v_train_120000"
    run_analysis_a(run_name)
    # run_save_to_pickle()
    # run_analysis_a()


if __name__ == "__main__":
    main()
