import argparse
import sys

from cache import save_to_pickle
from explain.genex.baseline_fns import label_predict
from explain.genex.load import load_as_simple_format
from models.transformer import hyperparams

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")
arg_parser.add_argument("--model_path", help="Your model path.")
arg_parser.add_argument("--method_name", )


def run(args):
    hp = hyperparams.HPGenEx()
    save_name = "{}_labels".format(args.data_name)
    data = load_as_simple_format(args.data_name)
    labels = label_predict(hp, data, args.model_path)

    save_to_pickle(labels, save_name)


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
