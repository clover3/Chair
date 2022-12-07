from collections import OrderedDict, Counter, defaultdict
from typing import List, Callable, Iterator

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from misc_lib import pause_hook, SuccessCounter
from trainer_v2.per_project.cip.cip_common import split_into_two, Comparison
from trainer_v2.per_project.cip.precomputed_cip import get_cip_pred_splits_iter


def iter_comparison() -> Iterator[Comparison]:
    todo = get_cip_pred_splits_iter()
    for split, itr, src_size in todo:
        if split == "train":
            yield from itr


def main():
    suc_counters = defaultdict(SuccessCounter)
    for comp in iter_comparison():
        n_try = len(comp.ts_pred_probs)
        full_pred = np.argmax(comp.full_pred_probs)

        label_counter = Counter()
        for i in range(n_try):
            st, ed = comp.ts_input_info_list[i]
            probs = comp.ts_pred_probs[i]
            local_d, global_d = probs
            ts_global_pred = np.argmax(global_d)
            label_counter[ts_global_pred] += 1
            suc_counters['ts_diff'].add(ts_global_pred != full_pred)
            suc_counters['ts_fail'].add(ts_global_pred != full_pred and full_pred == comp.label)

        majority, n_major = label_counter.most_common(1)[0]
        suc_counters['major_diff'].add(majority != full_pred)
        suc_counters['major_fail'].add(majority != full_pred and full_pred == comp.label)


    for key, sc in suc_counters.items():
        print(key, sc.get_suc_prob())


if __name__ == "__main__":
    main()