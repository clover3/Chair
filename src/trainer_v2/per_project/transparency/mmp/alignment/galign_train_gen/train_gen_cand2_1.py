import random
from cpath import output_path

from typing import List, Tuple
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.alignment.tt_datagen import get_encode_fn, \
    read_score_generate_term_pair_score_tfrecord
from misc_lib import path_join, group_by, get_first, ceil_divide
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def generate_train_data_inner(
        save_name, term_gain: List[Tuple[str, str, float]]):
    grouped = group_by(term_gain, get_first)
    q_term_list = list(grouped.keys())
    random.shuffle(q_term_list)
    n_item = len(q_term_list)
    n_fold = 10
    print(f"Total of {n_item} items")
    size_per_block = ceil_divide(n_item, n_fold)
    encode_fn = get_encode_fn()
    for fold_idx in range(n_fold):
        st = fold_idx * size_per_block
        ed = st + size_per_block
        q_term_list_split = q_term_list[st:ed]
        data = []
        for k in q_term_list_split:
            data.extend(grouped[k])
        save_path = path_join(output_path, "msmarco", "passage", "galign_v2", save_name, str(fold_idx))
        print(f"Fold {fold_idx} has {len(data)} items from {len(q_term_list_split)} q_terms")
        write_records_w_encode_fn(save_path, encode_fn, data)


def filter_tailing_sbword(table):
    def is_tailing(term):
        return term[:2] == "##"
    out_table = []
    for qt, dt, score in table:
        if is_tailing(qt) or is_tailing(dt):
            pass
        else:
            out_table.append((qt, dt, score))

    print(f"Filter out tailing subword: {len(table)} -> {len(out_table)}")
    return out_table


def main2():
    ph = get_cand2_1_path_helper()
    score_path = ph.per_pair_candidates.fidelity_table_path
    save_name = "cand2_1"
    term_gain = read_term_pair_table(score_path)
    term_gain_filtered = filter_tailing_sbword(term_gain)
    generate_train_data_inner(save_name, term_gain_filtered)


if __name__ == "__main__":
    main2()
