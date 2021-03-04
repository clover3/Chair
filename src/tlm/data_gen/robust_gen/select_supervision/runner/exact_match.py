import argparse
import os
import sys

from cache import load_pickle_from
from cpath import at_output_dir
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.robust_gen.select_supervision.gen_by_exact_match import generate_selected_training_data, \
    get_score_fn_functor

arg_parser = argparse.ArgumentParser(description='')


def main():
    target_data_idx = int(sys.argv[1])
    max_seq_length = int(sys.argv[2])
    max_seg = int(sys.argv[3])
    info_path = os.path.join(job_man_dir,
                             "robust_w_data_id_desc_info_pickle",
                             "{}".format(target_data_idx))

    info = load_pickle_from(info_path)
    save_dir_path = at_output_dir("robust_seg_sel", "exact_match{}_{}".format(max_seq_length, max_seg))
    exist_or_mkdir(save_dir_path)
    get_score_fn = get_score_fn_functor()
    generate_selected_training_data(info, max_seq_length, save_dir_path, get_score_fn, max_seg)


if __name__ == "__main__":
    main()
