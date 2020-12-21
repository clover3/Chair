import os
from collections import Counter
from typing import List, Dict

from arg.pers_evidence.get_lm import get_query_lms
from arg.pers_evidence.runner.get_candidate_dict import get_candidate_full_text
from arg.qck.decl import QCKCandidate
from arg.qck.topic_lm.ranker import rank_with_query_lm
from cpath import data_path
from evals.mean_average_precision import get_map
from evals.parse import load_qrels_flat
from evals.types import QRelsFlat


def main():
    split = "train"
    print("get query lms")
    query_lms: Dict[str, Counter] = get_query_lms(split)
    candidate_dict: Dict[str, List[QCKCandidate]] = get_candidate_full_text(split)
    q_ranked_list = rank_with_query_lm(query_lms, candidate_dict)
    qrel_path = os.path.join(data_path, "perspective", "evidence_qrel.txt")
    qrels: QRelsFlat = load_qrels_flat(qrel_path)
    score = get_map(q_ranked_list, qrels)
    print(score)


if __name__ == "__main__":
    main()
