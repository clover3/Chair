import argparse
import sys

from arg.qck.save_to_trec_form import save_to_common_path

parser = argparse.ArgumentParser(description='')


parser.add_argument("--pred_path")
parser.add_argument("--info_path")
parser.add_argument("--run_name")
parser.add_argument("--input_type", default="qck")
parser.add_argument("--max_entry", default=100)
parser.add_argument("--combine_strategy", default="avg_then_doc_max")
parser.add_argument("--score_type", default="softmax")
parser.add_argument("--shuffle_sort", default=False)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    save_to_common_path(args.pred_path,
                        args.info_path,
                        args.run_name,
                        args.input_type,
                        int(args.max_entry),
                        args.combine_strategy,
                        args.score_type,
                        args.shuffle_sort
    )