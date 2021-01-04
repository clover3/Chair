

from typing import Dict
from typing import List

from arg.qck.decl import QKUnit, QCKQuery, QCKCandidate
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator, QCKCandidateI
from arg.qck.qck_worker import QCKWorker
from arg.robust.qc_common import get_candidate_all_passage_w_samping
from cache import load_from_pickle, save_to_pickle, load_cache
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from evals.parse import load_qrels_structured


def main():
    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
    judgement = load_qrels_structured(qrel_path)

    def is_correct(query: QCKQuery, candidate: QCKCandidate):
        qid = query.query_id
        doc_id = candidate.id
        if qid not in judgement:
            return 0
        d = judgement[qid]
        label = 1 if doc_id in d and d[doc_id] > 0 else 0
        return label

    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate_filtered")

    candidate_dict = load_cache("candidate_for_robust_qck_7")
    if candidate_dict is None:
        candidate_dict: \
            Dict[str, List[QCKCandidateI]] = get_candidate_all_passage_w_samping()
        save_to_pickle(candidate_dict, "candidate_for_robust_qck_7")

    generator = QCKInstanceGenerator(candidate_dict, is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qck_10"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
