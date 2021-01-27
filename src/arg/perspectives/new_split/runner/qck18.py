from typing import List, Dict

from arg.perspectives.new_split.common import get_qck_candidate_for_split, split_name2
from arg.perspectives.new_split.runner.qck17 import qck_gen_w_ranked_list
from arg.qck.decl import QCKCandidate
from cache import load_from_pickle


def main():
    job_name = "qck18"
    split = "train"
    qks = load_from_pickle("pc_qk3_filtered_rel_{}".format(split))
    qck_candidates_dict: Dict[str, List[QCKCandidate]] = get_qck_candidate_for_split(split_name2, split)
    qck_gen_w_ranked_list(job_name,
                          qks,
                          qck_candidates_dict,
                          split)


if __name__ == "__main__":
    main()