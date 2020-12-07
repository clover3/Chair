import os
from typing import Dict
from typing import List

from arg.qck.decl import QKUnit, QCKQuery, QCKCandidate
from arg.qck.qck_worker import QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator, QCKCandidateI
from arg.robust.qc_common import get_candidate_all_passage_w_samping
from cache import load_from_pickle, save_to_pickle, load_cache
from cpath import data_path
from data_generator.data_parser.robust2 import load_qrel
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def main():
    qrel_path = os.path.join(data_path, "robust", "qrels.rob04.txt")
    judgement = load_qrel(qrel_path)

    def is_correct(query: QCKQuery, candidate: QCKCandidate):
        qid = query.query_id
        doc_part_id = candidate.id
        doc_id = "_".join(doc_part_id.split("_")[:-1])
        if qid not in judgement:
            return 0
        d = judgement[qid]
        if doc_id in d:
            return d[doc_id]
        else:
            return 0

    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_wiki_qk_candidate")

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
    job_name = "robust_qck_8"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()