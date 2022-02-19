from typing import Dict

import numpy as np

from cache import save_to_pickle
from misc_lib import tprint
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype_2X_v_train_200000, load_q_emb_qtype
from tlm.qtype.analysis_qde.analysis_a import cluster_avg_embeddings


def main():
    tprint("Loading pickle...")
    emb_d = load_q_emb_qtype_2X_v_train_200000()
    tuple_list = [(k, v) for k, v in emb_d.items()]
    cluster_avg_embeddings(tuple_list)


def main_2X():
    tprint("Loading pickle...")
    run_name = "qtype_2Y_v_train_120000"
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype(run_name)
    tuple_list = [(k, v) for k, v in q_embedding_d.items()]
    cluster_labels = cluster_avg_embeddings(tuple_list)
    save_to_pickle(cluster_labels, "{}_clusters".format(run_name))


def other_t():
    tprint("Loading pickle...")
    run_name = "qtype_2Y_v_train_120000"
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype(run_name)
    tuple_list = [(k, v) for k, v in q_embedding_d.items()]
    cluster_labels = cluster_avg_embeddings(tuple_list)
    save_to_pickle(cluster_labels, "{}_clusters_2".format(run_name))


if __name__ == "__main__":
    other_t()