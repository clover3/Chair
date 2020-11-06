import argparse
import sys

from arg.qck.save_to_trec_form import save_to_common_path

parser = argparse.ArgumentParser(description='')


parser.add_argument("--pred_path")
parser.add_argument("--info_path")
parser.add_argument("--run_name")
parser.add_argument("--input_type", default="qck")
parser.add_argument("--max_entry", default=-1)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    save_to_common_path(args.pred_path,
                        args.info_path,
                        args.run_name,
                        "qck",
                        50,
                        "avg"
    )