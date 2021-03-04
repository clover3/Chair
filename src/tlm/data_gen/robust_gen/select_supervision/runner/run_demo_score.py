import argparse
import os
import sys

from cache import load_pickle_from
from cpath import at_output_dir
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.robust_gen.select_supervision.gen_by_exact_match import generate_selected_training_data, \
    get_score_fn_functor, demo_score

arg_parser = argparse.ArgumentParser(description='')


def main():
    target_data_idx = int(sys.argv[1])
    info_path = os.path.join(job_man_dir,
                             "robust_w_data_id_desc_info_pickle",
                             "{}".format(target_data_idx))
    max_seq_length = 512
    info = load_pickle_from(info_path)
    demo_score(info, max_seq_length)


if __name__ == "__main__":
    main()