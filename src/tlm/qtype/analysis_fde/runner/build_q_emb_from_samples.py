import functools
import pickle
from collections import defaultdict
from typing import Dict

import numpy as np

from cache import save_to_pickle, load_from_pickle
from tlm.qtype.analysis_fde.q_emb_tool import apply_merge
from tlm.qtype.analysis_fde.sample_tool import enum_sample_entries


def build_save_q_emb_from_samples(run_name):
    q_embedding_dict = defaultdict(list)
    enum_entries_fn = functools.partial(enum_sample_entries, run_name)
    num_jobs = 37
    n_test = 0
    for i in range(num_jobs):
        try:
            print("JOB", i)
            for e, info, func_span_rep in enum_entries_fn(i):
                k = func_span_rep
                q_embedding_dict[k].append(e.qtype_weights_qe)
        except FileNotFoundError as e:
            print(e)
        except pickle.UnpicklingError as e:
            print(e)
    q_embedding_dict_merged: Dict[str, np.array] = apply_merge(q_embedding_dict)
    save_name = run_name + "_q_embedding"
    save_to_pickle(q_embedding_dict_merged, save_name)
    print("{} tested".format(n_test))


def build_save_bias_from_samples(run_name):
    q_bias_d = defaultdict(list)
    enum_entries_fn = functools.partial(enum_sample_entries, run_name)
    num_jobs = 37
    n_test = 0
    for i in range(num_jobs):
        try:
            print("JOB", i)
            for e, info, func_span_rep in enum_entries_fn(i):
                k = func_span_rep
                q_bias_d[k].append(e.q_bias)
        except FileNotFoundError as e:
            print(e)
        except pickle.UnpicklingError as e:
            print(e)
    q_bias_d_ex = {k: [np.array([v_i]) for v_i in v] for k, v in q_bias_d.items()}
    q_bias_d_merged: Dict[str, np.array] = apply_merge(q_bias_d_ex)
    q_bias_d_merged = {k: v[0] for k, v in q_bias_d_merged.items()}
    save_name = run_name + "_q_bias"
    save_to_pickle(q_bias_d_merged, save_name)
    print("{} tested".format(n_test))


def load_q_emb_qtype_2X_v_train_200000() -> Dict[str, np.array]:
    return load_from_pickle("qtype_2X_v_train_200000_q_embedding")


def load_q_emb_qtype(run_name) -> Dict[str, np.array]:
    return load_from_pickle("{}_q_embedding".format(run_name))


def load_q_bias(run_name) -> Dict[str, np.array]:
    return load_from_pickle("{}_q_bias".format(run_name))


def build_for_qtype_2X_v_train_200000():
    run_name = "qtype_2X_v_train_200000"
    build_save_q_emb_from_samples(run_name)


def main():
    run_name = "qtype_2Y_v_train_120000"
    build_save_bias_from_samples(run_name)
    # build_save_q_emb_from_samples(run_name)




if __name__ == "__main__":
    main()