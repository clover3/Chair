

from typing import Dict
from typing import List

from arg.qck.decl import QKUnit, QCKQuery, QCKCandidate
from arg.qck.qck_worker import QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator, QCKCandidateI
from arg.robust.qc_common import load_candidate_all_passage
from cache import load_from_pickle, save_to_pickle, load_cache
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def main():

    def is_correct(query: QCKQuery, candidate: QCKCandidate):
        return 0

    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate_filtered")

    candidate_dict = load_cache("candidate_for_robust_qck_10_predict")
    if candidate_dict is None:
        candidate_dict: \
            Dict[str, List[QCKCandidateI]] = load_candidate_all_passage(256)
        save_to_pickle(candidate_dict, "candidate_for_robust_qck_10_predict")

    generator = QCKInstanceGenerator(candidate_dict, is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qck_10_predict"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
