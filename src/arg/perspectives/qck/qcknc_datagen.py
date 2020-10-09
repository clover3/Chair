from typing import Dict, List, Tuple

from arg.perspectives.eval_caches import get_eval_candidate_as_pids, get_eval_candidates_1k
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.qck.decl import QCKCandidate, QCKQuery
from list_lib import lmap


def cid_pid_format_to_qck(candidate_pers):
    candidate_dict: Dict[str, List[QCKCandidate]] = dict()
    for cid, candidate_pids in candidate_pers:
        candidate_dict[str(cid)] = \
            lmap(lambda pid: QCKCandidate(str(pid), perspective_getter(pid)), candidate_pids)
    return candidate_dict


def get_eval_candidates_as_qck(split) -> Dict[str, List[QCKCandidate]]:
    candidate_pers: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids(split)
    return cid_pid_format_to_qck(candidate_pers)


def is_correct_factory():
    gold = get_claim_perspective_id_dict()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> int:
        pid_cluster = gold[int(query.query_id)]
        return int(any([int(candidate.id) in cluster for cluster in pid_cluster]))
    return is_correct


def get_eval_candidates_1k_as_qck(split) -> Dict[str, List[QCKCandidate]]:
    cid_dict_format: List[Tuple[int, List[Dict]]] = get_eval_candidates_1k(split)

    def convert(e) -> Tuple[int, List[int]]:
        cid, p_list = e
        return cid, lmap(lambda p: p['pid'], p_list)

    cid_pid_format: List[Tuple[int, List[int]]] = lmap(convert, cid_dict_format)
    return cid_pid_format_to_qck(cid_pid_format)

