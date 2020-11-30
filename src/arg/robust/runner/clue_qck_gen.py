from typing import List

from arg.qck.decl import QKUnit
from arg.qck.qck_worker import QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from arg.robust.qc_common import load_candidate_head_as_doc
from arg.robust.qck_gen import QRel
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def main():
    qrel = QRel()
    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    candidate_dict = load_candidate_head_as_doc(256)
    generator = QCKInstanceGenerator(candidate_dict, qrel.is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qck_4"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()

