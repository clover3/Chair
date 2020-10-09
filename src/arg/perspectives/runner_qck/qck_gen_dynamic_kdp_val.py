from typing import List, Dict

from arg.perspectives.eval_caches import get_extended_eval_candidate_as_qck
from arg.perspectives.eval_helper_qck import get_is_correct_fn
from arg.perspectives.load import load_claims_for_sub_split
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.qck.decl import QCKCandidate
from arg.qck.dynamic_kdp.qck_generator import QCKGenDynamicKDP


def get_qck_gen_dynamic_kdp():
    split = "train"
    candidate_d: Dict[str, List[QCKCandidate]] = get_extended_eval_candidate_as_qck(split)

    train2_claims = load_claims_for_sub_split("val")

    target_qids = list([str(c['cId']) for c in train2_claims ])
    queries = get_qck_queries(split)
    queries = list([q for q in queries if q.query_id in target_qids])
    return QCKGenDynamicKDP(queries, candidate_d, get_is_correct_fn())


if __name__ == "__main__":
    get_qck_gen_dynamic_kdp()