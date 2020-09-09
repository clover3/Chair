import argparse
import json
import sys
from typing import List, Dict

from arg.perspectives.eval_caches import get_eval_candidates_from_pickle
from arg.perspectives.load import load_dev_claim_ids, get_claims_from_ids
from arg.perspectives.ppnc.cppnc_payload import make_cppnc_problem
from arg.perspectives.ppnc.encode_common import encode_two_inputs
from cache import load_from_pickle

parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--prediction_path")
parser.add_argument("--save_name")
parser.add_argument("--config_path")

def main():
    args = parser.parse_args(sys.argv[1:])
    prediction_path = args.prediction_path
    data_id_info: Dict = load_from_pickle("pc_dev_passage_payload_info")
    save_name = args.save_name

    d_ids = list(load_dev_claim_ids())

    dev_claims: List[Dict] = get_claims_from_ids(d_ids)
    candidate_perspectives: Dict[int, List[Dict]] = dict(get_eval_candidates_from_pickle("dev"))
    config = json.load(open(args.config_path, "r"))
    print(config)
    make_cppnc_problem(prediction_path, data_id_info,
                       dev_claims,
                       candidate_perspectives,
                       config,
                       save_name,
                       encode_two_inputs
                       )


if __name__ == "__main__":
    main()

