import os
from collections import defaultdict

from cache import load_list_from_jsonl
from cpath import output_path
from data_generator.tokenizer_wo_tf import ids_to_text, get_tokenizer
from misc_lib import pause_hook, SuccessCounter
from trainer_v2.per_project.cip.cip_common import split_into_two
from trainer_v2.per_project.cip.precomputed_cip import get_cip_pred_splits_iter
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    split, itr, _ = get_cip_pred_splits_iter()[0]
    save_path = os.path.join(output_path, "nlits", "nli_cip2_0_train_val_scores")
    items = load_list_from_jsonl(save_path, lambda x: x)
    suc_counters = defaultdict(SuccessCounter)

    for comparison, scores in zip(itr, items):
        fail_probs = np.array(scores)[:, 1]
        rank = np.argsort(fail_probs)
        top_i = rank[0]
        ts_probs = comparison.ts_pred_probs[top_i]
        local_d, global_d = ts_probs
        ts_global_pred = np.argmax(global_d)
        full_pred = np.argmax(comparison.full_pred_probs)
        if full_pred != ts_global_pred and full_pred == comparison.label:
            suc_counters['bad_seg_rate'].suc()
        else:
            suc_counters['bad_seg_rate'].fail()

    for key, suc_counter in suc_counters.items():
        print(key, suc_counter.get_suc_prob())


if __name__ == "__main__":
    main()