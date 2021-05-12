import argparse
import sys

from tlm.data_gen.robust_gen.select_supervision.read_score import generate_selected_training_data_loop, \
    generate_selected_training_data
from tlm.data_gen.robust_gen.select_supervision.read_score_ablation import generate_selected_training_data_ablation

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--score_dir", )
arg_parser.add_argument("--info_dir", )
arg_parser.add_argument("--max_seq_length", )
arg_parser.add_argument("--save_dir", )
arg_parser.add_argument("--split_no", )


def main():
    args = arg_parser.parse_args(sys.argv[1:])
    generate_selected_training_data_loop(int(args.split_no),
                                         args.score_dir,
                                         args.info_dir,
                                         int(args.max_seq_length),
                                         args.save_dir,
                                         generate_selected_training_data_ablation("always")
                                         )


if __name__ == "__main__":
    main()
