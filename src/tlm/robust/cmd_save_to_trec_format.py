import argparse
import sys

from tlm.robust.save_to_trec_format import save_to_trec_format


def main():
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--pred_path")
    parser.add_argument("--payload_type")
    parser.add_argument("--data_id")
    parser.add_argument("--num_candidate")
    parser.add_argument("--run_name")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args(sys.argv[1:])

    save_to_trec_format(
        args.pred_path,
        args.payload_type,
        args.data_id,
        int(args.num_candidate),
        args.run_name,
        args.save_path)


if __name__ == "__main__":
    main()