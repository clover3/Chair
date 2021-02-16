import argparse
import sys
from typing import List

import numpy as np

from cache import save_to_pickle
from data_generator.shared_setting import BertNLI
from explain.genex.baseline_fns import baseline_predict
from explain.genex.load import load_as_simple_format
from explain.nli_gradient_baselines import nli_attribution_predict
from models.transformer import hyperparams

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")
arg_parser.add_argument("--model_path", help="Your model path.")
arg_parser.add_argument("--method_name", )


def run(args):
    hp = hyperparams.HPGenEx()
    nli_setting = BertNLI()

    if args.method_name in ['deletion_seq', "random", 'idf', 'deletion', 'LIME',
                            'term_deletion', 'replace_token', 'term_replace']:
        predictor = baseline_predict
    elif args.method_name in ["elrp", "deeplift", "saliency", "grad*input", "intgrad"]:
        predictor = nli_attribution_predict
    else:
        raise Exception("method_name={} is not in the known method list.".format(args.method_name))

    save_name = "{}_{}".format(args.data_name, args.method_name)
    data = load_as_simple_format(args.data_name)
    explains: List[np.array] = predictor(hp, nli_setting, data, args.method_name, args.model_path)

    save_to_pickle(explains, save_name)


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
