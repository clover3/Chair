import functools
import pickle
from collections import defaultdict, Counter
from typing import Dict

import numpy as np

from cache import save_to_pickle, load_from_pickle
from tlm.qtype.analysis_fde.sample_tool import enum_sample_entries


def merge_vectors_by_frequency(embedding_list):
    def get_vector_diff(v1, v2):
        mag = (sum(np.abs(v1)) + sum(np.abs(v2))) / 2
        error = np.sum(np.abs(v1 - v2))
        rel_error = error / mag
        return error, rel_error


    def sig(embedding):
        return np.sum(np.abs(embedding))
    embedding_list.sort(key=sig)
    prev_v = None
    v_idx = 0
    v_list = []
    cnt = Counter()
    for e in embedding_list:
        if prev_v is not None:
            error, rel_error = get_vector_diff(prev_v, e)
            if rel_error < 1e-2:
                cnt[v_idx] += 1
            else:
                prev_v = e
                v_idx += 1
                v_list.append(e)
                cnt[v_idx] += 1
        else:
            prev_v = e
            v_list.append(prev_v)
            cnt[v_idx] += 1
    if len(cnt) > 1:
        raise ValueError

    for key, _ in cnt.most_common(1):
        return v_list[key]

    raise Exception()


def apply_merge(func_str_to_vector) -> Dict[str, np.array]:
    out_d = {}
    for k, v in func_str_to_vector.items():
        try:
            out_d[k] = merge_vectors_by_frequency(v)
        except ValueError:
            print("ValueError at {}".format(k))

    return out_d


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


def load_q_emb_qtype_2Y_v_train_120000() -> Dict[str, np.array]:
    return load_from_pickle("qtype_2Y_v_train_120000_q_embedding")


def main():
    run_name = "qtype_2Y_v_train_120000"
    build_save_q_emb_from_samples(run_name)


if __name__ == "__main__":
    main()