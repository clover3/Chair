import pickle
from typing import Tuple

from arg.qck.dvp_to_correctness import dvp_to_correctness
from arg.util import load_run_config
from list_lib import dict_key_map


def main():
    config = load_run_config()
    dvp_list = pickle.load(open(config['dvp_path'], "rb"))
    dvp_to_correctness_dict = dvp_to_correctness(dvp_list, config)

    def convert_key(key: Tuple[str, Tuple[str, int]]):
        qid, (doc_id, doc_idx) = key
        return qid, "{}_{}".format(doc_id, doc_idx)

    save_dict = dict_key_map(convert_key, dvp_to_correctness_dict)

    pickle.dump(save_dict, open(config['save_path'], "wb"))


if __name__ == "__main__":
    main()
