from typing import Dict
from typing import List

from arg.qck.decl import QKUnit, QCKQuery, QCKCandidate
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator, QCKCandidateI
from arg.qck.qck_worker import QCKWorker
from arg.robust.qc_common import load_candidate_all_passage
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from trec.qrel_parse import load_qrels_structured


def main():
    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
    judgement = load_qrels_structured(qrel_path)

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

    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    candidate_dict: Dict[str, List[QCKCandidateI]] = \
        load_candidate_all_passage(256)
    generator = QCKInstanceGenerator(candidate_dict, is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qck_5"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()

