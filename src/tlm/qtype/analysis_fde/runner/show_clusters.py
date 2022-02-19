import os
from collections import defaultdict
from typing import Dict

import numpy as np

from cache import load_pickle_from, load_from_pickle
from cpath import output_path
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype, load_q_bias


def run_analysis_a(run_name):
    job_id = 0
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype(run_name)
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    save_path = os.path.join(output_path, "qtype", run_name + '_sample', str(job_id))
    qtype_entries, query_info_dict = load_pickle_from(save_path)
    cluster = load_from_pickle("{}_clusters".format(run_name))

    cluster_id_to_idx = defaultdict(list)
    for idx, cluster_id in enumerate(cluster):
        cluster_id_to_idx[cluster_id].append(idx)


    func_span_list = []
    for idx, key in enumerate(q_embedding_d.keys()):
        func_span_list.append(key)

    for cluster_id, indices in cluster_id_to_idx.items():
        print()
        for idx in indices:
            func_span = func_span_list[idx]
            print(cluster_id, func_span)


def main():
    run_name = "qtype_2Y_v_train_120000"
    run_analysis_a(run_name)


if __name__ == "__main__":
    main()