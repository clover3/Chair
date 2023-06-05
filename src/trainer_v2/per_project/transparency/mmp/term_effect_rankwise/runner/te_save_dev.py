import sys
import time
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

import h5py

from cache import load_list_from_jsonl
from list_lib import pairzip
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base


def save_te_to_hdf5(h5: h5py.File, prefix: str, te: TermEffectPerQuery):
    indices, values = zip(*te.changes)
    d = {
        'target_scores': te.target_scores,
        'base_scores': te.base_scores,
        'indices': indices,
        'values': values
    }
    for k, v in d.items():
        key = f"{prefix}_{k}"
        h5.create_dataset(key, data=v)


def measure_time():
    q_term = "nucleus"
    d_term = "the"

    job_no = 0
    st = time.time()
    te_list = load_list_from_jsonl(get_te_save_path_base(q_term, d_term, job_no), TermEffectPerQuery.from_json)
    ed = time.time()
    print("{} items".format(len(te_list)))
    print("jsonl : {}".format(ed - st))

    save_path = sys.argv[1]
    st = time.time()
    te_list = load_te_from_h5(save_path)
    ed = time.time()
    print("{} items".format(len(te_list)))
    print("h5 : {}".format(ed - st))


def load_te_from_h5(save_path):
    output = []
    f = h5py.File(save_path, "r")
    try:
        for i in range(10000):
            def get(k):
                key = f"{k}_{i}"
                return f[key]

            target_scores = get('target_scores')
            base_scores = get('base_scores')
            indices = get('indices')
            values = get('values')
            changes = pairzip(indices, values)
            te = TermEffectPerQuery(target_scores, base_scores, changes)
            output.append(te)
    except KeyError:
        pass
    return output


def main():
    q_term = "nucleus"
    d_term = "the"

    job_no = 0
    te_list = load_list_from_jsonl(get_te_save_path_base(q_term, d_term, job_no), TermEffectPerQuery.from_json)

    save_path = sys.argv[1]

    h5 = h5py.File(save_path, "w")

    acc: Dict[str, List[List]] = {}
    for i, te in enumerate(te_list):
        indices, values = zip(*te.changes)
        d = {
            'target_scores': te.target_scores,
            'base_scores': te.base_scores,
            'indices': list(indices),
            'values': list(values)
        }
        for k, v in d.items():
            if k == 'indices':
                dtype = 'i2'
            else:
                dtype = 'f2'
            key = f"{k}_{i}"
            ret = h5.create_dataset(key, data=v, dtype=dtype)
    #
    #         if k not in acc:
    #             acc[k] = []
    #
    #         if len(v) < seq_len:
    #             pad_len = seq_len - len(v)
    #             v = v + [-1] * pad_len
    #         acc[k].append(v)
    #
    # for k, v in acc.items():
    #     if k == 'indices':
    #         dtype = 'i2'
    #     else:
    #         dtype = 'f2'
    #     ret = h5.create_dataset(k, data=v, dtype=dtype)
    #     print(k, ret)



if __name__ == "__main__":
    measure_time()