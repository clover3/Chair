import argparse
import sys

from cpath import at_output_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.robust_gen.select_supervision.gen_by_exact_match import get_score_fn_functor, \
    generate_selected_training_data_w_json
from tlm.estimator_output_reader import load_combine_info_jsons

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--info_path", help="Your input file.")
arg_parser.add_argument("--target_data", help="Your input file.")
arg_parser.add_argument("--max_seq_length", help="Your input file.")
arg_parser.add_argument("--max_seg", help="Your input file.")


def main():
    args = arg_parser.parse_args(sys.argv[1:])
    target_data_idx = int(args.target_data)
    max_seq_length = int(args.max_seq_length)
    max_seg = int(args.max_seg)
    info_path = args.info_path
    info = load_combine_info_jsons(info_path)
    save_dir_path = at_output_dir("robust_seg_sel", "exact_match{}_{}".format(max_seq_length, max_seg))
    exist_or_mkdir(save_dir_path)
    get_score_fn = get_score_fn_functor()
    generate_selected_training_data_w_json(info, max_seq_length, save_dir_path, get_score_fn, max_seg)


if __name__ == "__main__":
    main()
