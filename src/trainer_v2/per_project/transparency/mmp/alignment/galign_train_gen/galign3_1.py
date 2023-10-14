import json
import math
import random
from cpath import output_path

from typing import List, Tuple

from table_lib import tsv_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from misc_lib import path_join, group_by, get_first, ceil_divide
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.alignment.tt_datagen import get_pairwise_encode_fn
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper, \
    get_cand2_2_list_path_helper, get_cand2_2_path_helper
from typing import List, Iterable, Callable, Dict, Tuple, Set


def sample_pairs(term_gain_per_query: List[Tuple[str, str, int]])\
        -> Iterable[Tuple[str, str, str]]:
    pos_term_list = [d_term for q_term, d_term, label in term_gain_per_query if label]
    neg_term_list = [d_term for q_term, d_term, label in term_gain_per_query if not label]
    random.shuffle(neg_term_list)

    if not term_gain_per_query:
        return
    if not neg_term_list or not pos_term_list:
        return

    n_neg = len(neg_term_list)
    pos_term_list = pos_term_list[:n_neg]

    q_term = term_gain_per_query[0][0]
    for i, pos in enumerate(pos_term_list):
        neg = neg_term_list[i]
        yield q_term, pos, neg


def filter_classify(
        term_gains: Iterable[Tuple[str, str, str]],
):
    for q_term, d_term, score_s in term_gains:
        score = float(score_s)
        if score > 0.2:
            yield q_term, d_term, 1
        elif score < 0.05:
            yield q_term, d_term, 0


def generate_tfrecord(
        term_gain: List[Tuple[str, str, int]],
        tfrecord_save_dir,
        tsv_save_dir,
):
    grouped = group_by(term_gain, get_first)
    q_term_list = list(grouped.keys())
    random.shuffle(q_term_list)
    n_item = len(q_term_list)
    n_fold = 10
    size_per_block = ceil_divide(n_item, n_fold)
    encode_fn = get_pairwise_encode_fn()
    for fold_idx in range(n_fold):
        st = fold_idx * size_per_block
        ed = st + size_per_block
        q_term_list_split = q_term_list[st:ed]
        data = []
        for k in q_term_list_split:
            per_query_entries = sample_pairs(grouped[k])
            data.extend(per_query_entries)

        tfrecord_save_path = path_join(tfrecord_save_dir, str(fold_idx))
        print(f"Fold {fold_idx} has {len(data)} items from {len(q_term_list_split)} q_terms")
        write_records_w_encode_fn(tfrecord_save_path, encode_fn, data)

        tsv_save_path = path_join(tsv_save_dir, str(fold_idx) + ".tsv")
        save_entries = []
        for k in q_term_list_split:
            save_entries.extend(grouped[k])
        save_tsv(save_entries, tsv_save_path)


def main():
    ph = get_cand2_2_path_helper()
    itr: Iterable[Tuple[str, str, str]] = tsv_iter(ph.per_pair_candidates.fidelity_table_path)
    itr2: List[Tuple[str, str, int]] = list(filter_classify(itr))
    save_name = "galign3"
    tfrecord_save_dir = path_join(output_path, "msmarco", "passage", "galign_v2", save_name)
    tsv_save_dir = path_join(output_path, "msmarco", "passage", "galign_v2", "galign3_tsv")
    generate_tfrecord(itr2, tfrecord_save_dir, tsv_save_dir)


if __name__ == "__main__":
    main()
