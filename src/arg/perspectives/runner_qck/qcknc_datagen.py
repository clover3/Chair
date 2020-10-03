from typing import List, Dict, Tuple

from arg.perspectives.eval_caches import get_eval_candidate_as_pids
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_train_val
from arg.qck.decl import QCKCandidate, QCKQuery
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from list_lib import lmap


def get_eval_candidates_as_qck(split) -> Dict[str, List[QCKCandidate]]:
    candidate_pers: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids(split)
    candidate_dict: Dict[str, List[QCKCandidate]] = dict()
    for cid, candidate_pids in candidate_pers:
        candidate_dict[str(cid)] = \
            lmap(lambda pid: QCKCandidate(str(pid), perspective_getter(pid)), candidate_pids)

    return candidate_dict


def is_correct_factory():
    gold = get_claim_perspective_id_dict()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> int:
        pid_cluster = gold[int(query.query_id)]
        return int(any([int(candidate.id) in cluster for cluster in pid_cluster]))
    return is_correct


def main():
    start_generate_jobs_for_train_val(QCKInstanceGenerator(get_eval_candidates_as_qck("train"),
                                                           is_correct_factory()),
                                      "qcknc")


if __name__ == "__main__":
    main()
