import os
from typing import Dict

import numpy as np

from cache import load_pickle_from
from cpath import output_path
from misc_lib import tprint
from tlm.qtype.analysis_fde.analysis_a import run_qtype_analysis_b
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype_2X_v_train_200000


def run_analysis_b():
    tprint("Loading pickle...")
    job_id = 0
    run_name = "qtype_2X_v_train_200000"
    save_path = os.path.join(output_path, "qtype", run_name + '_sample', str(job_id))
    qtype_entries, query_info_dict = load_pickle_from(save_path)
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype_2X_v_train_200000()
    run_qtype_analysis_b(qtype_entries, query_info_dict, q_embedding_d, True)


def main():
    run_analysis_b()


if __name__ == "__main__":
    main()
