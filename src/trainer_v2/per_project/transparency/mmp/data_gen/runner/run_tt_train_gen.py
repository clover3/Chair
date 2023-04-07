import os

from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample
from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_encode_fn_for_word_encoder


def main():
    save_dir = path_join("output", "msmarco", "passage")
    save_path = os.path.join(save_dir, "train_tt")
    itr = enum_pos_neg_sample(range(109))
    encode_fn = get_encode_fn_for_word_encoder()
    n_item = 1000 * 1000
    write_records_w_encode_fn(save_path, encode_fn, itr, n_item)


if __name__ == "__main__":
    main()