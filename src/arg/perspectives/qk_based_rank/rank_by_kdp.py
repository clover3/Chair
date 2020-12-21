import os
from collections import Counter
from typing import List, Dict

from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck
from arg.qck.decl import QCKCandidate, QKUnit
from arg.qck.topic_lm.kdp_to_lm import kdp_to_lm
from arg.qck.topic_lm.ranker import rank_with_query_lm
from cache import load_from_pickle
from cpath import data_path
from evals.mean_average_precision import get_map
from evals.parse import load_qrels_flat
from evals.types import QRelsFlat


def main():
    print("get query lms")
    split = "train"
    qk_candidate: List[QKUnit] = load_from_pickle("perspective_qk_candidate_filtered_{}".format(split))
    # qk_candidate: List[QKUnit] = load_from_pickle("perspective_qk_candidate_{}".format(split))
    candidate_dict: Dict[str, List[QCKCandidate]] = get_eval_candidates_as_qck("train")
    query_lms: Dict[str, Counter] = kdp_to_lm(qk_candidate)
    valid_qids: List[str] = list(query_lms.keys())
    target_candidate_dict = {}
    for k, c, in candidate_dict.items():
        if k in valid_qids:
            target_candidate_dict[k] = c
    alpha = 0.1
    q_ranked_list = rank_with_query_lm(query_lms, target_candidate_dict, 999, alpha)
    qrel_path = os.path.join(data_path, "perspective", "qrel.txt")
    qrels: QRelsFlat = load_qrels_flat(qrel_path)
    score = get_map(q_ranked_list, qrels)
    print(score)


if __name__ == "__main__":
    main()
