from typing import List, Dict

from arg.perspectives.eval_caches import get_extended_eval_candidate_as_qck
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_val
from arg.qck.decl import QCKCandidate, QCKQuery
from arg.qck.qcknc_datagen import QCKInstanceGenerator


def is_correct_factory():
    gold = get_claim_perspective_id_dict()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> int:
        pid_cluster = gold[int(query.query_id)]
        return int(any([int(candidate.id) in cluster for cluster in pid_cluster]))
    return is_correct


def main():
    split = "train"
    candidate_d: Dict[str, List[QCKCandidate]] = get_extended_eval_candidate_as_qck(split)
    start_generate_jobs_for_val(
        QCKInstanceGenerator(candidate_d,
                             is_correct_factory()),
        "qcknc_ex")


if __name__ == "__main__":
    main()
