import argparse
import sys
from typing import List, Dict

from arg.perspectives.eval_caches import get_eval_candidates_from_pickle
from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids
from arg.perspectives.ppnc.cppnc_payload import make_cppnc_dummy_problem
from arg.qck.encode_common import encode_two_inputs

parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--save_name")

def main():
    args = parser.parse_args(sys.argv[1:])
    save_name = args.save_name
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    candidate_perspectives: Dict[int, List[Dict]] = dict(get_eval_candidates_from_pickle("train"))
    make_cppnc_dummy_problem(claims,
                       candidate_perspectives,
                       save_name,
                       encode_two_inputs
                       )

if __name__ == "__main__":
    main()