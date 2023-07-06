import os
import sys
from collections import defaultdict

import numpy as np
from cache import load_pickle_from
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by, get_first, get_second
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict_empty
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign_v2
from trainer_v2.per_project.transparency.mmp.alignment.runner.galign2_predict import ThresholdConfig


def main():
    input_files = sys.argv[1]
    prediction_path = sys.argv[2]

    run_config = get_run_config_for_predict_empty()
    output = load_pickle_from(prediction_path)
    t_config = ThresholdConfig()

    label = output['label']
    majority_pred = np.zeros_like(label)
    gold: List[int] = np.reshape(label, [-1]).tolist()
    pred_score = output['align_probe']['all_concat']
    pred_label = np.less(0, pred_score).astype(int)
    t = read_galign_v2(input_files, run_config, t_config, False)
    t = t.unbatch()
    tokenizer = get_tokenizer()

    dataset_len = sum([1 for _ in t])

    def get_term(input_ids):
        return tokenizer.convert_ids_to_tokens(input_ids.numpy())[0]

    case_n = {
        (0, 0): 'tn',
        (1, 0): 'fp',
        (0, 1): 'fn',
        (1, 1): 'tp',
    }

    assert dataset_len == len(pred_label)
    summary_entry = []
    for item, pred, gold_l in zip(t, pred_label, gold):
        eval_case = case_n[pred[0], gold_l]
        e = get_term(item['q_term']), get_term(item['d_term']), eval_case
        summary_entry.append(e)

    for q_term, entries in group_by(summary_entry, get_first).items():
        print(f"Qterm: {q_term}")
        for eval_case, (ec_entries) in group_by(entries, lambda x: x[2]).items():
            s = ", ".join(map(get_second, ec_entries))
            print(eval_case, s)



if __name__ == "__main__":
    main()