import os
from collections import Counter
from typing import List, Dict

from arg.pers_evidence.runner.get_candidate_dict import get_candidate_full_text
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
    qk_candidate: List[QKUnit] = load_from_pickle("pc_evi_filtered_qk_{}".format(split))
    qk_candidate: List[QKUnit] = load_from_pickle("pc_evidence_qk".format(split))
    candidate_dict: Dict[str, List[QCKCandidate]] = get_candidate_full_text(split)
    query_lms: Dict[str, Counter] = kdp_to_lm(qk_candidate)
    valid_qids: List[str] = list(query_lms.keys())
    target_candidate_dict = {}
    for k, c, in candidate_dict.items():
        if k in valid_qids:
            target_candidate_dict[k] = c
    alpha = 0.5
    print("alpha", alpha)
    q_ranked_list = rank_with_query_lm(query_lms, target_candidate_dict, 100, alpha)
    qrel_path = os.path.join(data_path, "perspective", "evidence_qrel.txt")
    qrels: QRelsFlat = load_qrels_flat(qrel_path)
    score = get_map(q_ranked_list, qrels)
    print(score)


if __name__ == "__main__":
    main()
