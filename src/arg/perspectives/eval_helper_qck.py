from typing import List, Callable, Dict

from arg.perspectives.load import get_claim_perspective_id_dict2
from arg.qck.decl import QCKCandidate, QCKQuery


def get_is_correct_fn() -> Callable[[QCKQuery, QCKCandidate], bool]:
    cid_to_pids: Dict[int, List[int]] = get_claim_perspective_id_dict2()

    def is_correct_fn(q: QCKQuery, c: QCKCandidate) -> bool:
        return int(c.id) in cid_to_pids[int(q.query_id)]
    return is_correct_fn

