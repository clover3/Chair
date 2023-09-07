import os
import sys
from collections import defaultdict, Counter

import numpy as np
from cache import load_pickle_from
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from table_lib import tsv_iter
from evals.basic_func import get_acc_prec_recall_i
from list_lib import index_by_fn
from misc_lib import group_by, get_first, get_second
from tab_print import print_table
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict_empty
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign_v2
from trainer_v2.per_project.transparency.mmp.alignment.runner.galign2_predict import ThresholdConfig
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2.show_align_scores_comp_train_dev import \
    categorize_one
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import mmp_root


def load_tt_pred():
    input_files = path_join(output_path, "msmarco", "passage", "galign_v2", "cand2_2", "val")
    prediction_path = path_join(output_path, "tf_eval", "tta8_10000")
    run_config = get_run_config_for_predict_empty()
    output = load_pickle_from(prediction_path)
    t_config = ThresholdConfig()

    label = output['label']
    gold: List[int] = np.reshape(label, [-1]).tolist()
    pred_score = output['align_probe']['all_concat']
    t = read_galign_v2(input_files, run_config, t_config, False)
    t = t.unbatch()
    tokenizer = get_tokenizer()

    dataset_len = sum([1 for _ in t])

    def get_term(input_ids):
        return tokenizer.convert_ids_to_tokens(input_ids.numpy())[0]

    assert dataset_len == len(pred_score)
    summary_entry = []
    for item, score, gold_l in zip(t, pred_score, gold):
        e = get_term(item['q_term']), get_term(item['d_term']), score[0], gold_l
        summary_entry.append(e)

    return summary_entry


def main():
    tt_pred_entries: List[Tuple[str, str, float, int]] = load_tt_pred()
    train_score_path = path_join(
        mmp_root(), "align_scores", "candidate2.tsv")
    dev_score_path = path_join(
        mmp_root(), "align_scores", "candidate2_dev.tsv")

    def build_label(triplet_itr):
        d = {}
        for q_term, d_term, score in triplet_itr:
            d[q_term, d_term] = categorize_one(float(score))
        return d

    train_score_d = build_label(tsv_iter(train_score_path))
    dev_score_d = build_label(tsv_iter(dev_score_path))

    counter = Counter()
    per_src_label = defaultdict(list)
    for q_term, d_term, score, label in tt_pred_entries:
        try:
            key = q_term, d_term
            pred_label = score > 0
            train_label: str = train_score_d[key]
            dev_label: str = dev_score_d[key]

            label_s_to_i = {
                'POS': 1,
                "NEG": 0
            }
            if train_label != "ZERO" and dev_label != "ZERO":
                labels_d = {
                    'train': label_s_to_i[train_label],
                    'train_tf': int(label),
                    'dev': label_s_to_i[dev_label],
                    'pred': pred_label
                }
                for src_name, value in labels_d.items():
                    per_src_label[src_name].append(value)
        except KeyError:
            pass

    case_n = {
        (0, 0): 'tn',
        (1, 0): 'fp',
        (0, 1): 'fn',
        (1, 1): 'tp',
    }
    todo = [
        ("pred", "train"),
        ("pred", "dev"),
        ("train", "dev"),
    ]

    head = ["pred", "gold", "precision", "recall", "f1", 'accuracy']
    table = [head]
    for pred_src, gold_src in todo:
        pred = per_src_label[pred_src]
        gold = per_src_label[gold_src]
        metric_d = get_acc_prec_recall_i(pred, gold)

        row = [pred_src, gold_src,
               metric_d['precision'], metric_d['recall'], metric_d['f1'],
               metric_d['accuracy']]
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()
