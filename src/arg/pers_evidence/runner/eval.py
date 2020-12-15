from typing import List, Dict

from arg.pers_evidence.eval import get_precision_recall
from arg.pers_evidence.runner.get_candidate_dict import get_candidate_w_score
from arg.qck.decl import QCKCandidate, QCKQuery


def baseline_eval():
    split = "train"
    k = 30
    c_d: Dict[str, List[QCKCandidate]] = get_candidate_w_score(split, k)

    output_ranked_list = []
    for qid, ranked_list in c_d.items():
        output_ranked_list.append((QCKQuery(qid, ""), ranked_list))

    scores = get_precision_recall(output_ranked_list)
    print(scores)


if __name__ == "__main__":
    baseline_eval()