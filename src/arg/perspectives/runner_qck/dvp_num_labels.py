import pickle
from typing import Dict, Tuple

from arg.qck.dvp_to_correctness import dvp_to_correctness
from arg.util import load_run_config


def main():
    config = load_run_config()

    dvp_list = pickle.load(open(config['dvp_path'], "rb"))

    d: Dict[Tuple[str, Tuple[str, int]], bool] = dvp_to_correctness(dvp_list, config)

    num_true = sum([int(v) for v in d.values()])
    num_false = sum([1-int(v) for v in d.values()])

    print("true", num_true)
    print("false", num_false)
    print("true %", num_true / (num_true + num_false))

if __name__ == "__main__":
    main()