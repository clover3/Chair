from typing import List, Dict

from arg.perspectives.eval_caches import get_extended_eval_candidate_as_qck
from arg.perspectives.load import get_claim_perspective_id_dict2
from arg.qck.decl import QCKCandidate
from list_lib import lmap


def main():
    candidate: Dict[str, List[QCKCandidate]] = get_extended_eval_candidate_as_qck("train")
    cid_to_pids: Dict[int, List[int]] = get_claim_perspective_id_dict2()

    g_num_true = 0
    g_num_false = 0
    for qid, c_list in candidate.items():
        def is_correct_fn(c: QCKCandidate) -> bool:
            return int(c.id) in cid_to_pids[int(qid)]

        labels = lmap(is_correct_fn, c_list)

        num_true = sum(labels)
        num_false = len(labels) - num_true
        g_num_true += num_true
        g_num_false += num_false

    total = g_num_true + g_num_false
    print(g_num_true, g_num_false, g_num_true/total)


if __name__ == "__main__":
    main()